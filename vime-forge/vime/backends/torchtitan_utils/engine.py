"""TitanEngine: torchtitan component assembly plus the vime forward / training step.

Config comes from torchtitan's ConfigManager: the recipe named by ``--module`` / ``--config``
supplies the base, ``_titan_argv`` adds the values vime derives at runtime, and the script's dotted
flags come last and win.

``TitanEngine.__init__`` is ``Trainer.__init__``'s non-PP path minus the components vime owns, plus
vime's own steps, in upstream's order. The torchtitan commit is the one torchtitan-npu pins in its
``requirements.txt``; walk this constructor against that trainer.py before trusting a new version.
"""

import gc
import logging
import math
import os
import re

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor
from torchtitan.config import apply_overrides, ConfigManager, TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.tools import utils as tt_utils
from tqdm import tqdm

from vime.utils import accelerator
from vime.utils import perf_timeline as ptl
from vime.utils.memory_utils import clear_memory
from vime.utils.misc import load_function
from vime.utils.precision_guard import collect_torchtitan as pg_titan  # precision-guard
from vime.utils.routing_replay import apply_routing_replay_patch

from . import checkpoint as ckpt
from ._titan_compat import assert_model_matches_checkpoint, enable_fsdp_gradient_division
from .data import get_batch, restore_global_response_fields
from .loss import loss_function
from .quantization import apply_training_quantization, validate_quantized_model
from .update_weight.common import get_hf_per_tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _titan_argv(args):
    """torchtitan argv: the values vime derives, then the script's flags, which win on any overlap."""
    steps = (args.num_rollout or 1) * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    argv = [
        "--hf_assets_path", args.hf_checkpoint,  # HF weights + tokenizer
        "--training.steps", str(max(steps, 1)),  # LR-decay length = the run's optimizer steps
        # Cold-start seed, used only while the checkpoint directory holds no step. The script can
        # point it elsewhere; --checkpoint.no_initial_load_in_hf then selects the DCP format.
        "--checkpoint.initial_load_path", args.hf_checkpoint,
        "--checkpoint.initial_load_in_hf",
    ]
    # The checkpoint directory. An absolute folder replaces dump_folder in torchtitan's join, so
    # dump_folder keeps its own default and holds only what torchtitan writes there.
    if args.save:
        argv += ["--checkpoint.folder", os.path.abspath(args.save)]
    # The context bound is a vime arg (it also drives the rollout), so it is not repeated in the
    # script. local_batch_size is inert for vime, which builds its own micro-batches.
    max_ctx = int(getattr(args, "rollout_max_context_len", 0) or 0)
    if max_ctx:
        argv += ["--training.seq_len", str(max_ctx)]
    mbs = int(getattr(args, "micro_batch_size", 0) or 0)
    if mbs:
        argv += ["--training.local_batch_size", str(mbs)]
    return argv + list(args.titan_cli_overrides or [])


def build_titan_config(args):
    """Resolve the full ``Trainer.Config``: recipe + vime-derived values + script flags."""
    argv = _titan_argv(args)
    if not any(tok.split("=", 1)[0] in ("--module", "--config") for tok in argv):
        raise ValueError(
            "The torchtitan backend needs a recipe: pass --module and --config in the run script, "
            "e.g. --module vime.backends.torchtitan_utils.models.qwen3 --config vime_qwen3_4b "
            "(see vime/backends/torchtitan_utils/models/)."
        )
    return ConfigManager().parse_args(argv)


def _is_main_rank(parallel_dims) -> bool:
    """True only on the display/log rank: TP rank 0 and DP x CP rank 0."""
    if parallel_dims.tp > 1:
        tp_mesh = parallel_dims.get_optional_mesh(["tp"])
        if tp_mesh is not None and tp_mesh.get_local_rank() != 0:
            return False
    # Include CP so only one rank reports per TP group.
    loss_mesh = parallel_dims.get_optional_mesh("loss")
    loss_rank = loss_mesh.get_local_rank() if loss_mesh is not None else 0
    return loss_rank == 0


def _human_count(n: int) -> str:
    """Compact human-readable count (e.g. 4022468096 -> '4.02B')."""
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


def _log_parameter_counts(model_parts, parallel_dims, label: str) -> None:
    """Parameter summary of the parallelized model, main rank only. A DTensor's numel() is the
    global count under any sharding, expert parallelism included; to_local().numel() is this
    rank's share."""
    if not _is_main_rank(parallel_dims):
        return

    total = trainable = local = 0
    for mp in model_parts:
        for p in mp.parameters():
            n = p.numel()
            total += n
            if p.requires_grad:
                trainable += n
            local += p.to_local().numel() if isinstance(p, DTensor) else p.numel()
    pct = (100.0 * trainable / total) if total else 0.0
    logger.info(
        "\n"
        "================= torchtitan model: %s =================\n"
        "  total parameters        : %17s  (%s)\n"
        "  trainable parameters    : %17s  (%.1f%%)\n"
        "  local shard (this rank) : %17s  (%s)\n"
        "  parallelism             : dp_shard=%d dp_replicate=%d tp=%d cp=%d pp=%d ep=%d\n"
        "========================================================",
        label,
        f"{total:,}", _human_count(total),
        f"{trainable:,}", pct,
        f"{local:,}", _human_count(local),
        parallel_dims.dp_shard, parallel_dims.dp_replicate, parallel_dims.tp,
        parallel_dims.cp, parallel_dims.pp, parallel_dims.ep,
    )


class TitanEngine:
    """Owns the torchtitan components and runs vime's forward / training steps on them.

    The components are attributes, so the step methods take only what varies per call. Not a
    ``Configurable``: the config is the ``Trainer.Config`` the recipe produced.
    """

    @classmethod
    def initialize(cls, args, hf_config):
        """Assemble the components and restore from a training checkpoint.

        ``hf_config`` is the checkpoint's parsed config.json, loaded once by the actor.
        Returns (engine, loaded_rollout_id); -1 on a fresh start, so start_rollout_id is +1 of it.
        """
        assert not getattr(args, "moe_use_upcycling", False)  # todo: moe upcycling not supported
        engine = cls(build_titan_config(args), args=args, hf_config=hf_config)
        loaded_rollout_id = ckpt.load(engine.checkpointer, args)
        for model in engine.model_parts:
            validate_quantized_model(model, engine.config)
        # todo: critic output layer reinit not supported
        clear_memory()
        return engine, loaded_rollout_id

    def __init__(self, config, *, args, hf_config):
        self.args = args
        self.config = config
        self.hf_config = hf_config
        model_spec = apply_training_quantization(config)
        assert model_spec is not None, "model_spec must be set by the config_registry recipe"
        self.sequence_padding_multiple = 64 if model_spec.name == "qwen3_5" else 1

        self.device = tt_utils.get_local_device()
        tt_utils.device_module.set_device(self.device)

        # The process group already exists (vime's Ray train actor), so only the
        # mesh derivation is kept. world_size is a vime value; dp_shard=-1 is derived from it.
        self.parallel_dims = pd = ParallelDims.from_config(config.parallelism, world_size=args.world_size)
        pd.build_mesh()
        logger.info(
            "ParallelDims: dp_shard=%d dp_replicate=%d tp=%d cp=%d pp=%d ep=%d",
            pd.dp_shard, pd.dp_replicate, pd.tp, pd.cp, pd.pp, pd.ep,
        )

        # Sequence length divisibility under sequence/context parallelism. vime
        # packs its own micro-batches and pads each to a multiple of tp for the same reason.
        seq_len_divisor = (pd.tp if config.parallelism.enable_sequence_parallel else 1) * (
            2 * pd.cp if pd.cp > 1 else 1
        )
        if config.training.seq_len % seq_len_divisor != 0:
            raise ValueError(
                f"training.seq_len ({config.training.seq_len}) must be divisible by "
                f"{seq_len_divisor} for the configured sequence/context parallelism "
                "(vime sets it from --rollout-max-context-len)."
            )

        dist_utils.set_spmd_backend(config.parallelism.spmd_backend)
        if config.parallelism.spmd_backend == "spmd_types":
            # Activation-checkpoint backward needs the thread-local SPMD mesh.
            torch.autograd.set_multithreading_enabled(False)
        config.maybe_log()
        # torchtitan's determinism setup may rewrite this value. Restore the launcher value
        # because rollout workers use ``strict`` and a mismatch can deadlock weight sync.
        hccl_deterministic = os.environ.get("HCCL_DETERMINISTIC")
        dist_utils.set_determinism(
            pd, self.device, config.debug, distinct_seed_mesh_dims=["pp"]
        )
        if hccl_deterministic is not None:
            os.environ["HCCL_DETERMINISTIC"] = hccl_deterministic
        if bool(getattr(args, "rl_batch_invariant", False)):
            os.environ["HCCL_DETERMINISTIC"] = "strict"
            os.environ["LCCL_DETERMINISTIC"] = "1"
            if getattr(self.device, "type", None) == "npu":
                import torch_npu

                torch_npu.npu.set_deterministic_level(3)

        model_config = model_spec.model
        model_config.update_from_config(config=config)
        if config.override.imports:
            replacements = apply_overrides(config.override, config)
            assert replacements, (
                f"override.imports={config.override.imports} replaced nothing. apply_overrides only "
                "warns, and the first forward would then run upstream's implementation."
            )
            model_config = model_spec.model  # an override may have replaced the model config node

        # After the overrides, so the dumped config is the one that gets built.
        self._validate(config)
        config._validate_cuda_graphs()
        self.model_config = model_config

        # vime: config.json is the source of truth for the architecture; checked after the
        # overrides, so what is checked is what gets built.
        assert_model_matches_checkpoint(model_spec, hf_config)

        logger.info("Building %s %s", model_spec.name, model_spec.flavor)
        with (
            torch.device("meta"),
            tt_utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()
        model.verify_module_protocol()

        # vime: forward hooks on the router modules.
        if getattr(args, "use_routing_replay", False):
            apply_routing_replay_patch(model)

        model = model_spec.parallelize_fn(
            model,
            parallel_dims=pd,
            training=config.training,
            parallelism=config.parallelism,
            compile_config=config.compile,
            ac_config=config.activation_checkpoint,
            dump_folder=config.dump_folder,
        )

        # Upstream's cpu-offload variant of this is unreachable, _validate rejects
        # enable_cpu_offload.
        model.to_empty(device=tt_utils.device_type)
        with torch.no_grad():
            model.init_weights(buffer_device=None)
        model.train()
        self.model_parts = [model]

        # vime: before the optimizer is built, so frozen params get no update and no state slot.
        self._freeze_model_params()
        # After freezing, so the trainable count reflects the applied mask.
        _log_parameter_counts(self.model_parts, pd, f"{model_spec.name} {model_spec.flavor}")

        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(self.optimizers, self.model_parts, pd)
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers, training_steps=config.training.steps
        )

        # VIME's reducer loss is local to each CP rank and is scaled by dp_size before backward.
        # FSDP then averages only over the full DP group (dp_shard * dp_replicate); CP must not be
        # included here because CP contributions are additive and are combined by the CP gradient
        # collective. Multiplying by CP would under-scale gradients by CP, while a CP loss all-reduce
        # would over-scale them by CP. Keep this invariant explicit at the configuration boundary.
        self.dp_size = pd.dp_shard * pd.dp_replicate
        for mp in self.model_parts:
            # Set this even for dp_size == 1 so correctness does not depend on the upstream default.
            enable_fsdp_gradient_division(mp, self.dp_size)

        self.sd_adapter = (
            model_spec.state_dict_adapter(model_config, config.hf_assets_path)
            if model_spec.state_dict_adapter
            else None
        )
        self.checkpointer = config.checkpoint.build(
            dataloader=None,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={},  # the resume point is the rollout id, derived from the step directory
            sd_adapter=self.sd_adapter,
            base_folder=config.dump_folder,
        )
        # Weights are loaded by initialize(), which picks the path and the step.

        self.train_context = dist_utils.get_spmd_context(
            parallel_dims=pd, spmd_typechecking=False
        )

        # The model's compute precision (FSDP2 mixed_precision_param). Both the training forward and
        # the rollout (vLLM, which serves in this precision) run at it, so weight sync and HF export
        # cast to it. Derived once from the resolved config.
        self.compute_dtype = TORCH_DTYPE_MAP[config.training.mixed_precision_param]
        self._vocab_parallel = pd.tp > 1
        tp_mesh = pd.get_optional_mesh(["tp"]) if pd.tp > 1 else None
        self._tp_group = tp_mesh.get_group() if tp_mesh is not None else None

    # -- vime gates and vime-specific model surgery ------------------------------------------

    def _validate(self, config) -> None:
        """Reject configurations the backend does not implement, on the RESOLVED config.

        The vime-arg gates live in arguments.torchtitan_validate_args; everything that reads
        torchtitan config (parallelism degrees, dtype, context length) has to wait until here.
        """
        pd = self.parallel_dims
        assert pd.pp == 1, f"torchtitan backend does not support pipeline parallelism (pp={pd.pp})."
        backend = config.parallelism.spmd_backend
        if pd.cp > 1:
            assert config.parallelism.context_parallel_load_balancer is None, (
                "TorchTitan CP requires context_parallel_load_balancer=None: "
                "response spans assume contiguous sequence shards."
            )
            if (
                getattr(self.args, "use_routing_replay", False)
                or getattr(self.args, "use_rollout_routing_replay", False)
            ):
                assert pd.tp == 1, (
                    "CP routing replay currently requires TP=1; "
                    "TP/SP router layouts are not supported."
                )
        # Preserve the backend name used by the prepared Qwen3.5 TorchTitan revision;
        # TorchTitan validates which modes the installed version supports.
        assert backend in ("default", "partial_dtensor", "spmd_types"), (
            "VIME supports default, partial_dtensor and spmd_types; "
            f"got {backend}. full_dtensor activations are not supported by the RL loss."
        )
        if config.training.dtype != "float32":
            logger.warning(
                "training.dtype=%s (expected float32). bf16/fp16 master weights round away tiny "
                "lr*grad updates -> reward will not converge.",
                config.training.dtype,
            )
        assert config.training.mixed_precision_param == "bfloat16", (
            "torchtitan backend requires --training.mixed_precision_param bfloat16: the NPU fused "
            "attention kernel takes bf16/fp16 only, so a wider compute precision is silently "
            f"downcast inside attention (got {config.training.mixed_precision_param})."
        )
        assert not config.training.enable_cpu_offload, (
            "--training.enable_cpu_offload needs the CPU backend upstream's init_distributed would "
            "set up, but vime's train actor owns the process group. Large models use the "
            "swap-optimizer override instead."
        )
        if isinstance(config.activation_checkpoint, SelectiveAC.Config):
            logger.warning(
                "activation_checkpoint is SelectiveAC, which is numerically inconsistent for vime "
                "(no_grad old-forward + TP/DTensor); prefer FullAC or None in the recipe."
            )
        max_ctx = int(getattr(self.args, "rollout_max_context_len", 0) or 0)
        if max_ctx:
            assert config.training.seq_len >= max_ctx, (
                f"--training.seq_len ({config.training.seq_len}) must be >= rollout_max_context_len "
                f"({max_ctx}), otherwise the training forward pass exceeds the sequence boundary."
            )

    def _freeze_model_params(self) -> None:
        """Apply requires_grad by regex.

        --only-train-params-name-list: freeze everything, then unfreeze params matching any pattern.
        --freeze-params-name-list: freeze params matching any pattern (rest stay trainable).
        """
        only_train = getattr(self.args, "only_train_params_name_list", None)
        freeze = getattr(self.args, "freeze_params_name_list", None)
        for model in self.model_parts:
            if only_train:
                for name, param in model.named_parameters():
                    param.requires_grad = False
                    for pattern in only_train:
                        if re.search(pattern, name):
                            param.requires_grad = True
                            break
            if freeze:
                for name, param in model.named_parameters():
                    for pattern in freeze:
                        if re.search(pattern, name):
                            param.requires_grad = False
                            break

    # -- forward / training ------------------------------------------------------------------

    def _forward(self, batch):
        """One packed micro-batch through the model: flat [T] tokens -> logits [T, V].

        The batch dimension the pinned torchtitan expects is added and removed here and nowhere
        else, as views. ``get_attention_masks`` turns the position resets into whatever metadata
        type the configured attention backend takes. Under tp>1 the logits stay a vocab-sharded
        DTensor for the caller's to_local(); at tp=1 full_tensor() merges them.
        """
        tokens = batch["packed_ids"].unsqueeze(0)
        positions = batch["position_ids"].unsqueeze(0)
        return self._forward_model(tokens, positions).squeeze(0)

    def _forward_model(self, tokens, positions):
        """Run the model after constructing the attention metadata for packed tokens."""
        model = self.model_parts[0]
        build_attention_masks = getattr(model, "build_attention_masks", None)
        if build_attention_masks is not None:
            # RL targets are prepared separately; pass independent placeholder labels.
            cp_mesh = self.parallel_dims.get_mesh("cp") if self.parallel_dims.cp_enabled else None
            tokens, _, model_kwargs = build_attention_masks(
                tokens,
                torch.zeros_like(tokens),
                {"positions": positions},
                cp_mesh=cp_mesh,
                load_balancer_type=self.config.parallelism.context_parallel_load_balancer if cp_mesh is not None else None,
            )
        else:
            model_kwargs = {
                "attention_masks": model.get_attention_masks(positions),
                "positions": positions,
            }
            if self.parallel_dims.cp_enabled:
                tokens, _, model_kwargs = prepare_context_parallel_input(
                    tokens,
                    torch.zeros_like(tokens),
                    model_kwargs,
                    self.parallel_dims.get_mesh("cp"),
                    self.device,
                    self.config.parallelism.context_parallel_load_balancer,
                    getattr(self.config.parallelism, "context_parallel_ptrr_mask_key", None),
                )
        if self.config.parallelism.spmd_backend == "spmd_types":
            from torchtitan.distributed.spmd_types import annotate_input_spmd_types

            # Input preparation has already sliced CP inputs and positions exactly
            # once. Annotation assigns layouts without any further sharding.
            tokens, _, model_kwargs = annotate_input_spmd_types(
                self.parallel_dims,
                tokens,
                torch.zeros_like(tokens),
                model_kwargs,
            )
        logits = model(tokens, **model_kwargs)
        if isinstance(logits, DTensor):
            # CP leaves the sequence dimension sharded. Gathering it here
            # would make cp_response_index_spans address the wrong rows; TP keeps
            # its vocab-sharded DTensor for the loss helper to_local().
            if self.parallel_dims.cp > 1 and self.config.parallelism.spmd_backend == "spmd_types":
                logits = logits.to_local()
            elif not self._vocab_parallel:
                logits = logits.full_tensor()
        return logits

    def forward_only(self, per_microbatch_fn, data_iterator, num_microbatches, *, store_prefix=""):
        """no_grad micro-batch forward, collecting per-sample results (e.g. log_probs).

        num_microbatches is a per-step list; the DataIterator holds every micro-batch of every step
        in one flat schedule, so the whole rollout is consumed (sum over steps).
        """
        args = self.args
        for it in data_iterator:
            it.reset()

        for m in self.model_parts:  # inference mode: disable dropout / BN running stats
            m.eval()

        # Optional pre-hook (weight quantization / substitution and similar).
        if getattr(args, "custom_before_log_prob_hook_path", None):
            custom_before_log_prob_hook = load_function(args.custom_before_log_prob_hook_path)
            custom_before_log_prob_hook(args, self.model_parts, store_prefix)

        microbatch_pbar = tqdm(
            total=sum(num_microbatches),
            desc=f"{store_prefix.rstrip('_') or 'actor'} forward",
            unit="microbatch",
            dynamic_ncols=True,
            leave=False,
            disable=not _is_main_rank(self.parallel_dims),
        )

        collected = {}
        with torch.no_grad():
            for _ in range(sum(num_microbatches)):
                batch = get_batch(
                    args, data_iterator[0], self.parallel_dims, self.device,
                    sequence_padding_multiple=self.sequence_padding_multiple,
                )
                with self.train_context():
                    logits = self._forward(batch)
                    out = per_microbatch_fn(
                        args, logits, batch, self._tp_group,
                        with_entropy=False, vocab_parallel=self._vocab_parallel,
                    )
                if batch.get("cp_response_index_spans") is not None:
                    for key in out:
                        out[key] = restore_global_response_fields(
                            out[key], batch["response_offsets"],
                            batch["cp_response_index_spans"], batch.get("cp_group"),
                        )
                for k, v in out.items():
                    collected.setdefault(store_prefix + k, []).extend(v)
                microbatch_pbar.update(1)
        microbatch_pbar.close()

        for m in self.model_parts:  # restore training mode
            m.train()

        # Packing order differs from partition order; scatter each list back so that log_probs[i]
        # aligns with rewards[i] / loss_masks[i].
        origin_indices = [i for mb in data_iterator[0].micro_batch_indices for i in mb]
        n = len(origin_indices)
        for k, values in collected.items():
            reordered = [None] * n
            for value, origin_index in zip(values, origin_indices, strict=True):
                reordered[origin_index] = value
            collected[k] = reordered

        # todo: pp / multimodal; currently single PP direct collection
        return collected

    @ptl.perf_timer("optimizer_step")
    def train_one_step(
        self, rollout_id, step_id, role, data_iterator, num_microbatches, divisor,
        microbatch_pbar=None,
    ):
        """Single optimizer step over ``num_microbatches`` micro-batches.

        The DataIterator is NOT reset here, so consecutive steps consume their own slice of the flat
        schedule. loss_function applies `loss * dp_size / divisor` internally, so backward() takes no
        further coefficient. The update and the lr step are skipped when the grad norm is non-finite.

        Returns (grad_norm, metric_keys, metric_sums), aggregated across mb/DP by the caller.

        todo: MTP / PP combined-1f1b / multimodal not supported
        """
        args = self.args
        for m in self.model_parts:
            m.train()

        self.optimizers.zero_grad()

        # Optional custom hook before the optimizer step (weight surgery / external intervention).
        if getattr(args, "custom_megatron_before_train_step_hook_path", None):
            custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
            custom_before_train_step_hook(
                args, rollout_id, step_id, self.model_parts, self.optimizers, self.lr_schedulers
            )

        metric_keys, metric_sums = None, None

        for _ in range(num_microbatches):
            batch = get_batch(
                args, data_iterator[0], self.parallel_dims, self.device,
                sequence_padding_multiple=self.sequence_padding_multiple,
            )
            pg_titan.stamp_coord(args, batch, role=role, rollout_id=rollout_id, step_id=step_id)  # precision-guard

            # On entry to train_one_step, stage = replay_backward (set by actor.py).
            # The forward pass must use replay_forward (the hook calls pop_forward, advancing the
            # forward cursor). Otherwise the hook would advance the backward cursor, misaligning the
            # actual AC recompute.
            _rr_old_stage = None
            if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
                _rr_old_stage = os.environ.get("ROUTING_REPLAY_STAGE")
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"

            with self.train_context():
                logits = self._forward(batch)

                # After forward, restore stage = replay_backward so the AC recompute in backward
                # pops from the backward cursor.
                if _rr_old_stage is not None:
                    os.environ["ROUTING_REPLAY_STAGE"] = _rr_old_stage

                if getattr(args, "fine_grained_memory_cleanup", False):
                    clear_memory()

                # loss_function applies loss * dp_size / divisor internally; call backward() directly.
                loss, mb_metrics = loss_function(
                    args, logits, batch, self._tp_group,
                    vocab_parallel=self._vocab_parallel,
                    divisor=divisor,
                    dp_size=self.dp_size,
                )
                del logits  # lets backward free the [T,V] buffer as it goes
                if getattr(args, "fine_grained_memory_cleanup", False):
                    clear_memory()
                loss.backward()

            if metric_keys is None:
                metric_keys = list(mb_metrics.keys())
            mb_vec = torch.stack([mb_metrics[k] for k in metric_keys])
            metric_sums = mb_vec if metric_sums is None else metric_sums + mb_vec

            if microbatch_pbar is not None:
                microbatch_pbar.update(1)

        if getattr(args, "fine_grained_memory_cleanup", False):
            clear_memory()
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh(["pp"]),
            ep_enabled=self.parallel_dims.ep_enabled,
        )

        # CI: persist / compare the grad norm (what clip_grad_norm_ returns, i.e. before clipping)
        # against a saved reference.
        gn_value = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        if getattr(args, "ci_save_grad_norm", None) is not None:
            path = args.ci_save_grad_norm.format(role=role, rollout_id=rollout_id, step_id=step_id)
            torch.save(gn_value, path)
        elif getattr(args, "ci_load_grad_norm", None) is not None:
            path = args.ci_load_grad_norm.format(role=role, rollout_id=rollout_id, step_id=step_id)
            expected_grad_norm = torch.load(path)
            assert math.isclose(gn_value, expected_grad_norm, rel_tol=0.01, abs_tol=0.01), (
                f"grad norm mismatch: {gn_value} != {expected_grad_norm}"
            )

        # precision-guard: contains collectives, so it must not sit inside any rank branch.
        pg_titan.record_grad(  # precision-guard
            args, self.model_parts, self.parallel_dims, role=role, rollout_id=rollout_id, step_id=step_id,
        )

        if torch.isfinite(grad_norm):
            self.optimizers.step()
            # One .step() per optimizer step; the schedule spans training.steps of them.
            self.lr_schedulers.step()
        else:
            logger.warning("grad_norm is non-finite (%s), skipping update and lr step", grad_norm)

        pg_titan.record_optim(  # precision-guard
            args, self.optimizers, role=role, rollout_id=rollout_id, step_id=step_id,
            grad_norm=grad_norm, update_applied=bool(torch.isfinite(grad_norm)),
        )
        pg_titan.record_loss(  # precision-guard
            args, metric_keys, metric_sums, divisor,
            role=role, rollout_id=rollout_id, step_id=step_id, grad_norm=grad_norm,
        )
        pg_titan.record_weight(  # precision-guard
            args, self.model_parts, self.parallel_dims, role=role, rollout_id=rollout_id, step_id=step_id,
        )

        return grad_norm, metric_keys, metric_sums

    def train(self, rollout_id, data_iterator, num_microbatches, global_batch_sizes, role="actor"):
        """Training over a rollout of one or more optimizer steps.

        num_microbatches / global_batch_sizes are PER-STEP lists: one train_one_step per element,
        each consuming its own slice of the iterator's flat schedule.

        The divisor is that step's global token count (per-token loss, via a dp all-reduce) or its
        global_batch_size. The LAST step's reduced metrics are returned, along with its divisor so
        the caller reuses it without a redundant all-reduce.

        todo: MTP / PP / multimodal / offload_train not supported
        """
        args = self.args
        assert len(num_microbatches) == len(global_batch_sizes), (
            f"num_microbatches and global_batch_sizes must have the same length, "
            f"got {len(num_microbatches)} vs {len(global_batch_sizes)}"
        )

        # The token count must cover the same ranks the gradients are averaged over, which is
        # dp_size = dp_replicate * dp_shard, i.e. the "batch" mesh ("fsdp" is dp_shard * cp).
        dp_mesh = self.parallel_dims.get_optional_mesh(["batch"])
        per_token = bool(getattr(args, "calculate_per_token_loss", False))

        # Reset once for the whole rollout; train_one_step does not reset.
        for it in data_iterator:
            it.reset()

        for m in self.model_parts:
            m.train()

        if getattr(args, "reset_optimizer_states", False):
            _reset_optimizer_states(self.optimizers)

        # Manual GC: disable automatic collection so collection timing is aligned across ranks.
        manual_gc = getattr(args, "manual_gc", False)
        if manual_gc:
            gc.disable()
            gc.collect()

        num_steps_per_rollout = len(num_microbatches)
        rd = data_iterator[0].rollout_data
        grad_norm, metric_keys, metric_sums, divisor = float("nan"), None, None, None

        # Flat across steps; walked to map each step to its sample indices.
        mbi = data_iterator[0].micro_batch_indices
        mb_offset = 0

        microbatch_pbar = tqdm(
            total=sum(num_microbatches),
            desc=f"{role} train",
            unit="microbatch",
            dynamic_ncols=True,
            leave=False,
            disable=not _is_main_rank(self.parallel_dims),
        )

        for step_id in range(num_steps_per_rollout):
            step_num_mb = num_microbatches[step_id]
            if per_token:
                # This step's global token count; all ranks take part in the all-reduce.
                step_sample_ids = [i for mb in mbi[mb_offset : mb_offset + step_num_mb] for i in mb]
                local_tokens = torch.stack(
                    [rd["loss_masks"][i].float().sum().clamp(min=1.0) for i in step_sample_ids]
                ).sum()
                if dp_mesh is not None:
                    dist.all_reduce(local_tokens, group=dp_mesh.get_group())
                divisor = local_tokens.clamp(min=1.0)
            else:
                # Per-rollout mean: per-mb loss = reducer_sum * dp_size / step_global_batch_size.
                divisor = float(max(global_batch_sizes[step_id], 1))
            mb_offset += step_num_mb

            grad_norm, metric_keys, metric_sums = self.train_one_step(
                rollout_id, step_id, role, data_iterator, step_num_mb, divisor,
                microbatch_pbar=microbatch_pbar,
            )

            # Periodic manual collection (aligned timing across ranks).
            if manual_gc:
                interval = int(getattr(args, "manual_gc_interval", 0) or 0)
                if interval > 0 and (step_id + 1) % interval == 0:
                    gc.collect()

        microbatch_pbar.close()

        return grad_norm, metric_keys, metric_sums, divisor

    # -- checkpoint / weight export -----------------------------------------------------------

    def save_hf_model(self, rollout_id: int) -> None:
        """Export weights in HF safetensors format to args.save_hf.format(rollout_id=...).

        All ranks must call this: get_hf_per_tensor's full_tensor() is an all-gather over the
        FSDP/TP group. Only rank 0 writes, and a failure is a warning rather than a stop.
        """
        args = self.args
        if args.save_hf is None:
            return

        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        path = args.save_hf.format(rollout_id=rollout_id)

        try:
            extension = getattr(self.config, "extension", None)
            train_quantization = getattr(extension, "quantization", None)
            tensors = {
                name: t
                for name, t in get_hf_per_tensor(
                    self.model_parts, self.sd_adapter, self.device,
                    param_dtype=self.compute_dtype, parallel_dims=self.parallel_dims,
                    train_quantization=train_quantization,
                    model_type=self.config.model_spec.name,
                )()
            }
            if is_rank0:
                os.makedirs(path, exist_ok=True)
                save_file(
                    {k: v.contiguous().cpu() for k, v in tensors.items()},
                    os.path.join(path, "model.safetensors"),
                )
                self.hf_config.save_pretrained(path)
                logger.info("torchtitan HF weights saved @ rollout_id=%d -> %s", rollout_id, path)
        except Exception as e:  # noqa: BLE001
            if is_rank0:
                logger.error("Failed to save HF format weights: %s", e)


def _reset_optimizer_states(optimizer):
    """Zero AdamW exp_avg / exp_avg_sq / step across the torchtitan optimizer container."""
    # torchtitan OptimizersContainer is iterable over its wrapped torch optimizers.
    for opt in optimizer:
        for group in opt.param_groups:
            if "step" in group:
                group["step"] = 0
        for state in opt.state.values():
            if "step" in state:
                if isinstance(state["step"], torch.Tensor):
                    state["step"].zero_()
                else:
                    state["step"] = 0
            if "exp_avg" in state:
                state["exp_avg"].zero_()
            if "exp_avg_sq" in state:
                state["exp_avg_sq"].zero_()
