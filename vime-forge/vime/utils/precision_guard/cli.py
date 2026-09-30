"""Command-line arguments for the precision guard.

Kept here rather than in ``vime/utils/arguments.py`` so production code holds a single
``pg_cli.add_arguments(parser)`` line.
"""

from __future__ import annotations


def validate_args(args) -> None:
    """Called at the end of argument parsing; no-op when collection is off.

    Validating here makes an unusable site fail before training instead of surfacing as zero
    artifacts after a full run.
    """
    if not getattr(args, "precision_guard_dir", None):
        return
    from . import validate_sites  # noqa: PLC0415

    validate_sites(args)


def add_arguments(parser) -> None:
    """Register the --precision-guard-* options."""
    parser.add_argument(
        "--precision-guard-dir",
        type=str,
        default=None,
        help=(
            "Enable fingerprint collection and set the output root. Empty (default) makes "
            "every collection function return on its first line, at zero cost. Artifacts "
            "land at <dir>/{backend}/{role}/rollout_{id}/step_{id}/{site}.json."
        ),
    )
    parser.add_argument(
        "--precision-guard-sites",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Collect only these sites; all default sites when omitted. Valid values: "
            "logprob loss grad grad_norm optim_hparam weight optim_state hidden. "
            "hidden is opt-in and must be named explicitly."
        ),
    )
    parser.add_argument(
        "--precision-guard-master-weight",
        action="store_true",
        default=False,
        help=(
            "megatron only, ignored by torchtitan: collect the optimizer's fp32 master "
            "weights instead of the "
            "model's bf16 compute copy. Both backends keep fp32 masters, but megatron's "
            "model parameters are bf16, so comparing them directly mixes in bf16 rounding. "
            "At DP>1 the masters are DP-sharded and reassembled by a cross-DP gather."
        ),
    )
    parser.add_argument(
        "--precision-guard-hidden-extra",
        type=str,
        nargs="+",
        default=[],
        choices=["act_grad", "logits"],
        help=(
            "Extra quantities for the hidden site. act_grad records the activation gradient "
            "flowing back into each layer (agrad_NN); logits records the lm_head output with "
            "the vocab dimension gathered to full width, which costs a few GB per micro-batch."
        ),
    )
    parser.add_argument(
        "--precision-guard-dtype",
        type=str,
        default=None,
        choices=["float32", "bfloat16", "float16"],
        help=(
            "Cast per-parameter sites (weight/grad) to this precision before fingerprinting. "
            "Omitted (default) keeps the original precision. Cross-backend comparison needs "
            "bfloat16 so both sides match megatron's bf16 model weights; same-backend "
            "sensitivity work should keep the original, since bf16's ~4e-3 relative "
            "resolution quantizes small perturbations away. Each tensor's actual dtype is "
            "recorded in the artifact."
        ),
    )
    parser.add_argument(
        "--precision-guard-expert-limit",
        type=int,
        default=-1,
        help=(
            "Fingerprint only global MoE experts with ids below this limit for weight and "
            "grad sites. The default -1 collects all experts; positive values skip higher "
            "expert ids before EP all-gather."
        ),
    )
