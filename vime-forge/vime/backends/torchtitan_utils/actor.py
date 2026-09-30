"""TorchTitanTrainRayActor: the torchtitan training actor.

Implements the TrainRayActor contract on top of torchtitan components.
"""

import logging
import os
import random

import numpy as np
import ray
import torch
import torch._dynamo
import torch.distributed as dist
from torch_memory_saver import torch_memory_saver

from vime.observability import logging_utils
from vime.observability.profile_utils import TrainProfiler
from vime.observability.train_data_utils import save_debug_train_data
from vime.observability.timer import Timer, inverse_timer, timer, with_defer
from vime.ray.train_actor import TrainRayActor
from vime.utils import accelerator
from vime.utils.data import process_rollout_data
from vime.utils.distributed_utils import get_gloo_group, init_dp_gloo_group
from vime.utils.hf_config import load_hf_config
from vime.utils.memory_utils import clear_memory, print_memory
from vime.utils import perf_timeline as ptl
from vime.utils.misc import load_function
from vime.utils.precision_guard import attach as pg_attach  # precision-guard
from vime.utils.processing_utils import load_tokenizer
from vime.utils.routing_replay import RoutingReplay
from vime.utils.types import RolloutBatch

from . import checkpoint as ckpt
from .data import build_packed_microbatch, get_data_iterator, log_perf_data, log_rollout_data
from .engine import TitanEngine
from .loss import compute_advantages_and_returns, get_log_probs_and_entropy
from .update_weight.common import get_hf_per_tensor
from .update_weight.update_weight_from_distributed import UpdateWeightFromDistributed
from .update_weight.update_weight_from_tensor import UpdateWeightFromTensor
from .weights_backuper import TitanWeightsBackuper

logger = logging.getLogger(__name__)


def _current_lrs(optimizer) -> list[float]:
    """Read the current lr from the optimizer's param_groups.

    torchtitan's LRSchedulersContainer is a plain container of LambdaLRs and has no
    ``get_last_lr``. LambdaLR writes ``param_groups[i]["lr"]``, which is the source of truth.
    """
    lrs: list[float] = []
    for opt in getattr(optimizer, "optimizers", [optimizer]):
        for group in opt.param_groups:
            lrs.append(float(group["lr"]))
    return lrs


def _needs_actor_snapshot(*, with_ref: bool, keep_old_actor: bool) -> bool:
    """Whether training needs a CPU actor snapshot for model switching."""
    return with_ref or keep_old_actor


class TorchTitanTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        if args.debug_rollout_only:
            self.args = args
            return 0

        super().init(args, role, with_ref, with_opd_teacher)

        if accelerator.device_type() == "npu":
            # Disable torch.compile on the TRAIN side only: ppo_utils (compute_policy_loss / entropy /
            # kl) is @torch.compile and miscompiles on torch_npu inductor. Set it here (in the train
            # actor process) rather than via a job-wide TORCHDYNAMO_DISABLE env, so the vLLM rollout
            # workers, which rely on vLLM's own torch.compile (VLLM_COMPILE), are NOT affected.
            torch._dynamo.config.disable = True

        if dist.get_rank() == 0:
            logging_utils.init_tracking(args, primary=False, role=role)

        self.prof = TrainProfiler(args)
        self.published_policy_version = None

        # Serialize tokenizer / hf_config loading to avoid concurrent write races.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = load_hf_config(args.hf_checkpoint)
                self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        dist.barrier(group=get_gloo_group())

        if args.offload_train and (margin := getattr(args, "train_memory_margin_bytes", 2147483648)) > 0:
            logger.info("Set torch_memory_saver.memory_margin_bytes to %d", margin)
            torch_memory_saver.memory_margin_bytes = margin

        # Keep a CPU-backed copy of TorchTitan's FSDP shards and optimizer state.
        # ``sleep`` then releases the NPU allocations while rollout owns the colocated cards.
        tms_region_ctx = None
        if args.offload_train and accelerator.device_type() == "npu":
            tms_region_ctx = torch_memory_saver.region(tag="training", enable_cpu_backup=True)
            tms_region_ctx.__enter__()
        self.engine, loaded_rollout_id = TitanEngine.initialize(args, self.hf_config)
        if tms_region_ctx is not None:
            tms_region_ctx.__exit__(None, None, None)
        pd = self.engine.parallel_dims

        # Build the DP-scoped gloo group now that the mesh exists (collective: all ranks reach here).
        # Lets _gather_log_data gather metrics exactly across DP even at tp>1 (see init_dp_gloo_group).
        init_dp_gloo_group(pd)
        # TitanEngine.initialize returns loaded_rollout_id: fresh start = -1, resumed at step N = N.
        # Training starts from the next step.
        start_rollout_id = loaded_rollout_id + 1

        self.train_parallel_config = self._get_parallel_config()

        pg_attach(self)  # precision-guard

        if role == "critic":
            if args.offload_train:
                self.sleep()
            return start_rollout_id

        # Weight backuper: one CPU copy of this rank's shards per tag. It works on the tensors
        # rather than on state_dict because the hook-produced keys cannot be written back.
        self.weights_backuper = TitanWeightsBackuper(self.engine.model_parts)
        self._active_model_tag: str | None = "actor"
        if _needs_actor_snapshot(
            with_ref=with_ref,
            keep_old_actor=getattr(args, "keep_old_actor", False),
        ):
            self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        if with_opd_teacher:
            # todo: opd_teacher not supported
            pass

        if getattr(args, "keep_old_actor", False):
            # Initialize old_actor = actor (PPO ratio=1 on the first update; semantically
            # correct since old_actor is the "previous policy" and actor starts from the
            # same seed weights). Using copy instead of load_other_checkpoint avoids the
            # DCP-format limitation of load_other_checkpoint (only HF safetensors supported).
            self.weights_backuper.copy(src_tag="actor", dst_tag="old_actor")
            if args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

        # Loading ref/teacher above left their weights as the live model; restore actor so the
        # initial weight push to the rollout engines and the first forward use actor weights.
        if self._active_model_tag != "actor":
            self._switch_model("actor")

        if getattr(args, "vocab_size", None) is None:
            # Prefer HF config vocab_size (may include model-native padding); fall back to tokenizer.
            hf_vocab = getattr(self.hf_config, "vocab_size", None)
            args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size

        device = self.engine.device
        if args.colocate:
            update_weight_cls = UpdateWeightFromTensor
        else:
            update_weight_cls = UpdateWeightFromDistributed
        extension = getattr(self.engine.config, "extension", None)
        train_quantization = getattr(extension, "quantization", None)

        # Validate export configuration during initialization; the updater calls
        # the returned factory to read fresh weights on every synchronization.
        self.weight_updater = update_weight_cls(
            args,
            get_hf_per_tensor(
                self.engine.model_parts,
                self.engine.sd_adapter,
                device,
                param_dtype=self.engine.compute_dtype,
                parallel_dims=self.engine.parallel_dims,
                yield_experts_on_all_ranks=args.colocate,
                train_quantization=train_quantization,
                model_type=self.engine.config.model_spec.name,
            ),
        )

        clear_memory()

        if args.offload_train:
            self._switch_model("actor")
            self.sleep()

        self.rollout_data_postprocess = None
        if getattr(args, "rollout_data_postprocess_path", None) is not None:
            self.rollout_data_postprocess = load_function(args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train
        clear_memory(clear_host_memory=True)
        print_memory("before offload TorchTitan model")
        torch_memory_saver.pause()
        print_memory("after offload TorchTitan model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up TorchTitan model")
        torch_memory_saver.resume()
        clear_memory()
        print_memory("after wake_up TorchTitan model")

    @ptl.perf_consume
    @ptl.perf_timer("data_preprocess")
    def _get_rollout_data(self, rollout_data_ref) -> RolloutBatch:
        rollout_data = process_rollout_data(
            rollout_data_ref,
            self._dp_rank(),
            self.engine.dp_size,
        )

        device = self.engine.device

        rollout_data["tokens"] = [
            torch.tensor(t, dtype=torch.long, device=device) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            torch.tensor(t, dtype=torch.int, device=device) for t in rollout_data["loss_masks"]
        ]
        if "rollout_mask_sums" in rollout_data:
            rollout_data["rollout_mask_sums"] = torch.tensor(
                rollout_data["rollout_mask_sums"], dtype=torch.float32, device=device
            )

        if "multimodal_train_inputs" in rollout_data:
            # todo: multimodal not supported
            rollout_data["multimodal_train_inputs"] = [
                (
                    {
                        key: (
                            torch.from_numpy(v.copy()).to(device=device)
                            if isinstance(v, np.ndarray)
                            else v.to(device=device)
                        )
                        for key, v in mm_dict.items()
                    }
                    if mm_dict is not None
                    else None
                )
                for mm_dict in rollout_data["multimodal_train_inputs"]
            ]

        if getattr(self.args, "qkv_format", "thd") == "bshd":
            # todo: bshd max_seq_lens not supported
            pass

        for key in ["rollout_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            # CP log-prob fields are sliced by the CP data path; tensorize raw data directly.
            rollout_data[key] = [
                torch.tensor(log_prob, device=device, dtype=torch.float32)
                for log_prob in rollout_data[key]
            ]

        if "rollout_routed_experts" in rollout_data:
            # Convert to torch.Tensor (zero-copy), keep on CPU: fill_routing_replay's
            # RoutingReplay.record stores to CPU pinned memory, no GPU transfer needed.
            rollout_data["rollout_routed_experts"] = [
                torch.from_numpy(r) if isinstance(r, np.ndarray) else r
                for r in rollout_data["rollout_routed_experts"]
            ]

        return rollout_data

    def _switch_model(self, target_tag: str) -> None:
        if target_tag == self._active_model_tag:
            return
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def _dp_rank(self) -> int:
        # Local rank within the DP sub-mesh ("batch" = dp_replicate * dp_shard). When tp>1, all ranks
        # in the same TP group share the same position in it, so dp_rank and the inputs match across TP.
        dp_mesh = self.engine.parallel_dims.get_optional_mesh(["batch"])
        return dp_mesh.get_local_rank() if dp_mesh is not None else 0  # dp=1: no mesh -> single DP group -> dp_rank=0 (not global rank)

    def _get_parallel_config(self) -> dict:
        # This backend has no VPP (vpp_size=1, microbatch_group=1); cp comes from ParallelDims.
        return {
            "dp_size": self.engine.dp_size,
            "cp_size": self.engine.parallel_dims.cp,
            "vpp_size": 1,
            "microbatch_group_size_per_vp_stage": 1,
        }

    @ptl.perf_timer_prefix("log_probs")
    def compute_log_prob(
        self,
        data_iterator: list,
        num_microbatches: list,
        store_prefix: str = "",
    ) -> dict:
        """No-gradient forward pass that collects per-token log_probs (and optionally entropy).

        Wraps model.forward_only + loss.get_log_probs_and_entropy;
        result keys are prefixed with store_prefix ("ref_" or "").
        """
        with timer(f"{store_prefix}log_probs"):
            return self.engine.forward_only(
                get_log_probs_and_entropy,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
            )

    @ptl.perf_step
    @ptl.perf_timer("train_phase")
    def train(self, rollout_id: int, rollout_data_ref, external_data=None):
        """Training entry point: preprocess data, dispatch to critic/actor training, optionally offload sleep."""
        if self.args.debug_rollout_only:
            return None

        if self.args.offload_train:
            self.wake_up()  # todo: offload_train not supported

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        # Ensure rollout_mask_sums exists (loss reduction denominator);
        # falls back to per-sample mask sum when not provided by the rollout side.
        if "rollout_mask_sums" not in rollout_data:
            rollout_data["rollout_mask_sums"] = torch.tensor(
                [float(m.sum().item()) for m in rollout_data["loss_masks"]],
                dtype=torch.float32,
                device=self.engine.device,
            )

        if self.role == "critic":
            result = self.train_critic(rollout_id, rollout_data)  # todo: critic not supported
        else:
            self.train_actor(rollout_id, rollout_data, external_data=external_data)
            result = None

        if self.args.offload_train:
            del rollout_data
            self.sleep()  # todo: offload_train not supported

        return result

    @ptl.perf_timer("actor_train")
    def train_actor(self, rollout_id: int, rollout_data, external_data=None) -> None:
        """GRPO actor training step orchestration.

        Order: get_data_iterator -> ref log_prob (via _switch_model) -> old/cur log_prob (short-circuit or recompute)
        -> compute_advantages_and_returns -> rollout_data_postprocess hook -> log_rollout_data
        -> engine.train -> _log_train_metrics -> backup("actor") -> periodic backup("ref") -> log_perf_data.
        teacher / critic values branches are kept as todos.
        """
        # get_data_iterator / log_* / compute_advantages_and_returns are module-level imports
        # (no circular dependencies); the training step itself runs on self.engine.

        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                # Ref model forward: provides KL baseline ref_log_probs.
                if self.with_ref:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )

                # Teacher model forward: provides teacher_log_probs (OPD).
                if "teacher" in self.weights_backuper.backup_tags:
                    # todo: opd_teacher not supported
                    pass

                # Switch to actor (or old_actor) to compute old/cur log_probs.
                self._switch_model("old_actor" if getattr(self.args, "keep_old_actor", False) else "actor")

                # Short-circuit: when these hold, the separate old-logprob forward is skipped and
                # the loss reuses the freshly computed log_probs as old (ratio 1 on the single update).
                # Off-policy: reuse log_probs recorded during rollout sampling (skip recomputation).
                use_rollout_logprobs = getattr(self.args, "use_rollout_logprobs", False)
                get_mismatch_metrics = getattr(self.args, "get_mismatch_metrics", False)
                can_reuse_log_probs_in_loss = (
                    len(num_microbatches) == 1
                    and self.args.loss_type == "policy_loss"
                    and self.args.kl_coef == 0
                    and not use_rollout_logprobs
                    and not get_mismatch_metrics
                    and not getattr(self.args, "use_critic", False)
                    and not getattr(self.args, "keep_old_actor", False)
                    and not getattr(self.args, "use_opd", False)
                    and (not self.args.use_routing_replay or self.args.use_rollout_routing_replay)
                    and self.args.advantage_estimator != "gspo"
                )

                if (not use_rollout_logprobs or get_mismatch_metrics) and not can_reuse_log_probs_in_loss:
                    # The else branch (record) is unreachable today: arguments.py rejects
                    # use_routing_replay without use_rollout_routing_replay. It is kept for
                    # future use_routing_replay expansion (router-computed routing without
                    # rollout source); it will not be entered under current config.
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="",
                        )
                    )
                    # old_actor and actor train read the same routing data, each from the
                    # start, so the forward index goes back to 0 between them. The recorded
                    # indices and the backward index stay.
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                if getattr(self.args, "use_critic", False):
                    # todo: critic values not supported
                    pass

                # Switch back to actor weights before computing advantages (avoid staying on ref after the switch).
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # dp_group is the DP sub-mesh, for cross-DP advantage normalization.
                dp_mesh = self.engine.parallel_dims.get_optional_mesh(["batch"])
                dp_group = dp_mesh.get_group() if dp_mesh is not None else None

                span = ptl.start_perf_span("actor_compute_advantages", self.args)
                compute_advantages_and_returns(self.args, rollout_data, dp_group)
                ptl.end_perf_span(span)

            # Externally injected postprocess hook (agnostic; configured via rollout_data_postprocess_path).
            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args, rollout_id, rollout_data)

            log_rollout_data(rollout_id, self.args, rollout_data, self.engine.parallel_dims)

            # Training step: returns grad_norm/metric_keys/metric_sums/divisor for metric reporting.
            # divisor is computed by engine.train (includes per-token all-reduce); reused directly to avoid
            # redundant distributed communication.
            with timer("actor_train"):
                if self.args.use_routing_replay:
                    os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
                grad_norm, metric_keys, metric_sums, divisor = self.engine.train(
                    rollout_id,
                    data_iterator,
                    num_microbatches,
                    global_batch_sizes,
                    role=self.role,
                )

            self.prof.step(rollout_id=rollout_id)

        self._log_train_metrics(rollout_id, metric_keys, metric_sums, divisor, grad_norm, rollout_data)

        # The step is done: drop this step's recorded indices (the next step re-records) and
        # release their pinned memory. The stage goes back to "fallthrough" so a router forward
        # before the next step's setup does not pop from the emptied lists.
        if self.args.use_routing_replay:
            RoutingReplay.clear_all()
            os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"

        # Optionally dump the rollout data used for this training step (debug / replay).
        save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        # Refresh the CPU snapshot only when a later model switch needs it.
        if _needs_actor_snapshot(
            with_ref=self.with_ref,
            keep_old_actor=getattr(self.args, "keep_old_actor", False),
        ):
            span = ptl.start_perf_span("backup_weights", self.args)
            self.weights_backuper.backup("actor")
            ptl.end_perf_span(span)

        # Periodically refresh the ref snapshot (agnostic; controlled by ref_update_interval).
        if (
            getattr(self.args, "ref_update_interval", None) is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and self.with_ref
        ):
            with timer("ref_model_update"):
                logger.info("Updating ref model at rollout_id %d", rollout_id)
                self.weights_backuper.backup("ref")

        log_perf_data(rollout_id, self.args, self.engine.parallel_dims)

        # todo: critic values writeback not supported

    def train_critic(self, rollout_id: int, rollout_data):
        raise NotImplementedError  # todo: critic not supported

    @ptl.perf_timer("fill_routing_replay")
    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        """Load routing from rollout data into RoutingReplay instances.

        Each micro-batch stores [total_seq_len, topk], and the hook reshapes it to whatever the
        router returned. The indices are global expert ids, so EP>1 needs nothing here: the
        dispatcher routes them. Contiguous CP is supported; SP / VPP are out of scope.
        """
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data "
                "when use_rollout_routing_replay is set."
            )

        # apply_routing_replay_patch registers instances to all_routing_replays via __init__.
        replays = RoutingReplay.all_routing_replays
        num_layers = rollout_data["rollout_routed_experts"][0].shape[1]
        assert len(replays) == num_layers, (
            f"router count={len(replays)}, but routed_experts has {num_layers} layers"
        )

        for iterator in data_iterator:
            iterator.reset()

        num_experts = self.args.num_experts

        def pad_func(experts, pad):
            """Pad routing with ``arange % num_experts``: spreading the pad tokens over valid
            expert ids keeps them from piling onto expert 0 and skewing tokens_per_expert_E."""
            _, num_layers, topk = experts.shape
            pad_tensor = (
                torch.arange(pad * num_layers * topk, device=experts.device, dtype=experts.dtype)
                % num_experts
            ).reshape((pad, num_layers, topk))
            return torch.cat([experts, pad_tensor], dim=0)

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
            routed = batch["rollout_routed_experts"]
            tokens = batch["tokens"]

            assert len(routed) == len(tokens)
            for experts, toks in zip(routed, tokens, strict=False):
                assert experts.shape[0] == toks.shape[0] - 1, (
                    f"experts seq={experts.shape[0]}, tokens seq={toks.shape[0]}"
                )

            # padding: seq_len-1 → seq_len (pad one token; loss_mask will mask it out)
            padded = [pad_func(r, 1) for r in routed]

            # Concatenate along the token dimension: [total_seq_len, num_layers, topk],
            # where total_seq_len = sum(seq_len_i) = T of the packed router input.
            packed = torch.cat(padded, dim=0).to(torch.int64)

            pd = self.engine.parallel_dims
            if pd.cp > 1:
                from torchtitan.distributed.context_parallel import cp_shard

                # Match get_batch's global padding before sharding route input rows.
                packed_mb = build_packed_microbatch(
                    tokens, "cpu",
                    pad_to_multiple=int(getattr(self.args, "packed_seq_alignment", 1)) * pd.cp * max(int(pd.tp), 1),
                )
                packed = pad_func(packed, packed_mb["packed_ids"].numel() - packed.shape[0])
                (packed,), _ = cp_shard(
                    pd.get_mesh("cp"),
                    (packed.to(self.engine.device).unsqueeze(0),),
                    None,
                    load_balancer_type=self.engine.config.parallelism.context_parallel_load_balancer,
                    input_seq_dim=1,
                )
                packed = packed.squeeze(0)

            # Record per-layer routing.
            for layer_id in range(packed.shape[1]):
                layer_routed = packed[:, layer_id, :]
                replays[layer_id].record(layer_routed)

        del rollout_data["rollout_routed_experts"]

        for iterator in data_iterator:
            iterator.reset()

    def _log_train_metrics(self, rollout_id, keys, sums, divisor, grad_norm, data):
        """Sum across microbatches, all-reduce across DP x CP, divide by divisor, then log on rank 0.

        The loss terms are per-rollout averages over ``divisor`` (the step's global batch size, or
        its token count in per-token mode), and carry the last step's values. Advantages are logged
        on the rollout side instead. ⚠ The all_reduce runs on every rank; only rank 0 writes.
        """
        if keys is None:
            return
        loss_mesh = self.engine.parallel_dims.get_optional_mesh("loss")
        grp = loss_mesh.get_group() if loss_mesh is not None else None
        vals = sums.clone()
        if grp is not None:
            dist.all_reduce(vals, group=grp)  # sum CP-local numerators across DP x CP; divisor is already global
        div = divisor if torch.is_tensor(divisor) else float(divisor)
        reduced = {k: (vals[i] / div).item() for i, k in enumerate(keys)}
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        role_tag = "" if self.role == "actor" else f"{self.role}-"
        log_dict = {f"train/{role_tag}{k}": v for k, v in reduced.items()}
        log_dict[f"train/{role_tag}grad_norm"] = float(grad_norm)
        # Per-param-group lr.
        lrs = _current_lrs(self.engine.optimizers)
        for pg_id, lr in enumerate(lrs):
            log_dict[f"train/{role_tag}lr-pg_{pg_id}"] = float(lr)
        log_dict[f"train/{role_tag}global_batch_size"] = int(sum(data["global_batch_sizes"]))
        log_dict["train/step"] = int(rollout_id)

        # CI checks, gated on rollout_id == 0.
        if getattr(self.args, "ci_test", False):
            diff_key = f"train/{role_tag}train_rollout_logprob_abs_diff"
            if diff_key in log_dict:
                assert log_dict[diff_key] <= 0.1, f"{log_dict=}"
            if not getattr(self.args, "ci_disable_kl_checker", False):
                ppo_kl_key = f"train/{role_tag}ppo_kl"
                clipfrac_key = f"train/{role_tag}pg_clipfrac"
                if rollout_id == 0 and ppo_kl_key in log_dict and clipfrac_key in log_dict:
                    assert log_dict[ppo_kl_key] < 1e-8, f"{log_dict=}"
                kl_loss_key = f"train/{role_tag}kl_loss"
                if (
                    rollout_id == 0
                    and not getattr(self.args, "use_rollout_routing_replay", False)
                    and kl_loss_key in log_dict
                ):
                    assert log_dict[kl_loss_key] < 1e-8, f"{log_dict=}"

        logger.info(
            "train metrics @ rollout_id=%d: %s",
            rollout_id,
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in log_dict.items()},
        )
        logging_utils.log(self.args, log_dict, step_key="train/step")

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        # todo: async_save force_sync not supported
        if self.args.debug_rollout_only:
            return
        if self.args.offload_train:
            self.wake_up()  # todo: offload_train not supported
        ckpt.save(self.engine.checkpointer, self.args, rollout_id)
        if self.args.save_hf is not None and self.role == "actor":
            self.engine.save_hf_model(rollout_id)
        if self.args.offload_train:
            self.sleep()  # todo: offload_train not supported

    @timer
    @ptl.perf_policy_version
    @ptl.perf_timer("weight_sync")
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return
        # todo: use_fault_tolerance not supported
        (rollout_engines, rollout_engine_lock, num_new_engines,
         engine_gpu_counts, engine_gpu_offsets, engine_parallel_configs) = ray.get(
            self.rollout_manager.get_updatable_engines_and_lock.remote())
        # The NPU IPC sender reads device tensors. Restore the CPU-backed
        # TorchTitan shards only for the transfer, then offload them again
        # before vLLM recreates its KV cache.
        if self.args.offload_train:
            self.wake_up()
        if num_new_engines > 0:
            self.weight_updater.connect_rollout_engines(
                rollout_engines, rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
                engine_parallel_configs=engine_parallel_configs)
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())
        self.weight_updater.update_weights()
        self.published_policy_version = str(self.weight_updater.weight_version)
        if self.args.offload_train:
            self.sleep()
        if self.args.ci_test and len(rollout_engines) > 0:
            engine = random.choice(rollout_engines)
            engine_version = ray.get(engine.get_weight_version.remote())
            if str(engine_version) != str(self.weight_updater.weight_version):
                raise RuntimeError(
                    f"Weight version mismatch! Engine: {engine_version}, "
                    f"Updater: {self.weight_updater.weight_version}"
                )
        # keep_old_actor: rotate the policy snapshot queue after each weight push.
        # update_weights_interval==1: queue rotation (rollout_actor -> old_actor, actor -> rollout_actor).
        # update_weights_interval>1: old_actor = current actor (no rollout_actor needed).
        if getattr(self.args, "keep_old_actor", False):
            if self.args.update_weights_interval == 1:
                logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                self.weights_backuper.backup("rollout_actor")
            else:
                self.weights_backuper.backup("old_actor")

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        """Load weights from ``path`` into the live model, then snapshot them under model_tag.

        ``path`` is an HF safetensors directory; a DCP step directory works too once the caller
        passes from_hf=False."""
        # The load overwrites the live weights, so the actor needs a snapshot first.
        if model_tag != "actor" and "actor" not in self.weights_backuper.backup_tags:
            self.weights_backuper.backup("actor")
        ckpt.load_from(
            self.engine.checkpointer,
            path,
            model_only=True,
            from_hf=True,
            from_quantized=bool(
                getattr(self.engine.config.checkpoint, "initial_load_in_hf_quantized", False)
            ),
        )
        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    def connect_actor_critic(self, critic_group) -> None:
        raise NotImplementedError  # todo: critic not supported
