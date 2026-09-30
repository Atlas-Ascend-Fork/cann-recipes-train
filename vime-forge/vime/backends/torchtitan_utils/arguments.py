"""torchtitan backend argument parsing and validation.

Self-contained argparse rather than the megatron parser: reaching that parser imports
`vime.backends.megatron_utils`, which triggers `import mindspeed.megatron_adaptor` and patches
torch_npu, breaking a torchtitan+vllm stack (torch2.10, no megatron/mindspeed installed).

torchtitan_parse_args: vime common args -> vime operational args -> parse_known_args (leftovers
split into the torchtitan pass-through channel and dropped flags) -> megatron-free defaults ->
arch args backfilled from the HF config.
"""

import argparse
import logging

from vime.utils import accelerator

logger = logging.getLogger(__name__)

# torchtitan flags carrying no dot: ConfigManager's recipe selectors (torchtitan/config/manager.py).
_TITAN_BARE_FLAGS = ("module", "config")


def _safe_add(parser, *names, **kwargs):
    """Add an argument only if none of its option strings are already registered."""
    if any(n in parser._option_string_actions for n in names if n.startswith("-")):
        return
    parser.add_argument(*names, **kwargs)


def _add_vime_operational_args(parser):
    """vime-side knobs that vime's common parser does not register. What torchtitan owns arrives as
    dotted flags, and the model architecture is backfilled from the HF config."""
    _safe_add(parser, "--fp16", action="store_true", default=False)  # flips the bf16 default
    # Leave TorchTitan's --debug.batch_invariant in its dotted override channel.
    _safe_add(
        parser,
        "--rl-batch-invariant",
        action="store_true",
        default=False,
    )
    _safe_add(
        parser,
        "--fine-grained-memory-cleanup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect garbage and release device cache during training to reduce memory pressure.",
    )
    # Checkpoint loading. --load and --save come from the common parser: --save is the checkpoint
    # directory, --load names a different one to read from. These two say which step and how much.
    _safe_add(parser, "--load-step", type=int, default=None)  # default: the largest step present
    _safe_add(parser, "--load-mode", type=str, default="full", choices=["full", "weights"])
    _safe_add(parser, "--profile-step-start", type=int, default=None)
    _safe_add(parser, "--profile-step-end", type=int, default=None)
    _safe_add(parser, "--record-memory-history", action="store_true", default=False)
    _safe_add(parser, "--memory-snapshot-path", type=str, default=None)
    _safe_add(parser, "--tensorboard-dir", type=str, default=None)
    _safe_add(parser, "--use-pytorch-profiler", action="store_true", default=False)
    _safe_add(
        parser,
        "--packed-seq-alignment",
        type=int,
        default=1,
        help="Base packed sequence alignment; padding uses this value times CP times TP. "
        "Set it to the alignment required by the model kernels.",
    )
    return parser


def _split_titan_cli_overrides(unknown):
    """Split parse_known_args leftovers into (titan_argv, other).

    A token is torchtitan's iff its flag name carries a dot (``--optimizer.beta2``) or is in
    _TITAN_BARE_FLAGS; both ``--a.b v`` and ``--a.b=v`` survive verbatim. The caller discards
    ``other``.
    """
    dotted, other, i = [], [], 0
    while i < len(unknown):
        tok = unknown[i]
        name = tok[2:].split("=", 1)[0] if tok.startswith("--") else ""
        if "." in name or name in _TITAN_BARE_FLAGS:
            dotted.append(tok)
            if "=" not in tok and i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                dotted.append(unknown[i + 1])
                i += 2
            else:
                i += 1
        else:
            other.append(tok)
            i += 1
    return dotted, other


def torchtitan_parse_args(extra_args_provider=None, skip_hf_validate=False):
    """Self-contained torchtitan argument parsing. See module docstring."""
    parser = argparse.ArgumentParser(allow_abbrev=False)
    if extra_args_provider is not None:
        maybe = extra_args_provider(parser)
        if maybe is not None:
            parser = maybe
    _add_vime_operational_args(parser)
    args, unknown = parser.parse_known_args()
    args.titan_cli_overrides, _ = _split_titan_cli_overrides(unknown)

    # megatron-free defaults (mirrors _set_default_megatron_args).
    args.rank = 0
    if getattr(args, "critic_train_only", False):
        args.world_size = args.critic_num_nodes * args.critic_num_gpus_per_node
    else:
        args.world_size = args.actor_num_nodes * args.actor_num_gpus_per_node
    args.bf16 = not getattr(args, "fp16", False)

    # Sole provider of the megatron-named arch args that flops_utils / vllm_rollout read, so it runs
    # whenever a checkpoint is set (not gated on skip_hf_validate) and always leaves every attribute
    # defined. Driver-side, so the training and rollout actors inherit the same values.
    if getattr(args, "hf_checkpoint", None):
        try:
            from vime.utils.hf_config import load_hf_config

            from ._arch_fields import populate_vime_arch_args

            _hf = load_hf_config(args.hf_checkpoint)
            _hf = getattr(_hf, "text_config", _hf)  # unwrap multimodal wrapper
            _filled = populate_vime_arch_args(args, _hf)
            if _filled:
                logger.info(
                    "[arch-args] populated from HF config: %s",
                    ", ".join(f"{k}={v}" for k, v in _filled),
                )
        except AssertionError:
            raise  # strict arch mismatch -> surface loudly
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not populate arch args from HF config (%s); relying on script-provided args.", e
            )
    return args


def torchtitan_validate_args(args):
    """Reject unimplemented paths that would otherwise fall back silently to dense+GRPO handling.

    Only gates whose source is a vime arg live here; the ones reading torchtitan config
    (pp/cp degrees, context length, spmd backend) run in TitanEngine._validate against the
    resolved config.
    """
    assert accelerator.is_available() and accelerator.device_type() == "npu", (
        "torchtitan backend runs on Ascend NPU only; use --train-backend megatron elsewhere."
    )

    if args.packed_seq_alignment <= 0:
        raise ValueError("--packed-seq-alignment must be positive")

    if getattr(args, "qkv_format", None) == "bshd":  # megatron only
        raise AssertionError("torchtitan backend does not support qkv_format=bshd (megatron only).")

    # Only GRPO is wired in; another estimator would be silently treated as GRPO.
    est = getattr(args, "advantage_estimator", "grpo")
    assert est == "grpo", (
        f"torchtitan backend only supports --advantage-estimator grpo (current={est}). "
        "Other estimators' advantage/loss are not implemented in loss.policy_loss_function."
    )

    assert getattr(args, "balance_data", False), (
        "torchtitan backend requires --balance-data: ensures equal sample count across DP ranks, "
        "otherwise FSDP chunked-forward collectives mismatch and the loss normalization stops "
        "matching the global batch."
    )

    # The context bound has a single owner: TitanEngine turns it into --training.seq_len, which
    # sizes the RoPE cache. Left unset, the recipe's own (much smaller) default would silently cap
    # the model.
    assert getattr(args, "rollout_max_context_len", None), (
        "torchtitan backend requires --rollout-max-context-len: it is the context bound for the "
        "rollout AND the model's training.seq_len (RoPE cache size)."
    )

    # Naming a step or a mode says which checkpoint to read, so there has to be one to read from.
    if getattr(args, "load_step", None) is not None or getattr(args, "load_mode", "full") == "weights":
        assert args.load or args.save, (
            "--load-step / --load-mode need a checkpoint directory: pass --load, or --save when "
            "reading from the run's own directory."
        )

    # --colocate enables offload_train as a scheduling marker; the actor's sleep/wake calls are
    # no-ops there because vLLM's sleep mode owns rollout memory. Standalone offload is unsupported.
    assert not getattr(args, "offload_train", False) or getattr(args, "colocate", False), (
        "torchtitan backend does not support --offload-train (training memory offload; wake_up wrapping not implemented)."
    )

    # Flags that rewrite the loss/sampling path. use_rollout_logprobs and use_tis are implemented
    # and stay allowed.
    for flag in ("use_opd",):
        assert not getattr(args, flag, False), (
            f"torchtitan backend does not support --{flag.replace('_', '-')} "
            "(rewrites loss/sampling path, not implemented in loss.policy_loss_function)."
        )

    # use_rollout_routing_replay is implemented and implies use_routing_replay, so only the reverse
    # (train-time routing with no rollout source) is rejected.
    assert not (
        getattr(args, "use_routing_replay", False)
        and not getattr(args, "use_rollout_routing_replay", False)
    ), (
        "torchtitan backend: use_routing_replay without use_rollout_routing_replay "
        "(router computes routing at train time, no rollout source) is not yet "
        "implemented. Use --use-rollout-routing-replay instead."
    )

    # --use-tis / --use-rollout-logprobs mutual exclusion is enforced for every backend by
    # vime_validate_args, which runs first; not duplicated here.
