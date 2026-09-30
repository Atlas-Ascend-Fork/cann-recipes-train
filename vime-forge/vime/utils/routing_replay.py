import logging
import os

import torch

from vime.utils import accelerator

logger = logging.getLogger(__name__)


ROUTING_REPLAY = None
ORDERED_TOPK_CAPTURE_ROUTER = None


def set_routing_replay(replay):
    global ROUTING_REPLAY
    ROUTING_REPLAY = replay


def _set_ordered_topk_capture_router(router):
    global ORDERED_TOPK_CAPTURE_ROUTER
    ORDERED_TOPK_CAPTURE_ROUTER = router


def _capture_ordered_topk(top_indices):
    """Keep the current router's exact top-k order until its MoE combine."""
    router = ORDERED_TOPK_CAPTURE_ROUTER
    if router is not None:
        router._vime_ordered_topk_indices = top_indices


def consume_ordered_topk(module):
    """Return and release the current forward's ordered top-k indices."""
    return module.__dict__.pop("_vime_ordered_topk_indices", None)


def register_ordered_topk_capture(module):
    """Capture one forward's VLLM-compatible top-k order without R3."""
    if getattr(module, "_vime_ordered_topk_capture_registered", False):
        return

    def pre_forward_hook(patched_module, *args, **kwargs):
        del args, kwargs
        _set_ordered_topk_capture_router(patched_module)

    def forward_hook(patched_module, *args, **kwargs):
        del args, kwargs
        if ORDERED_TOPK_CAPTURE_ROUTER is patched_module:
            _set_ordered_topk_capture_router(None)

    module.register_forward_pre_hook(pre_forward_hook)
    module.register_forward_hook(forward_hook)
    module._vime_ordered_topk_capture_registered = True


def _compute_topk_for_current_router(
    old_compute_topk,
    scores,
    topk,
    num_groups=None,
    group_topk=None,
):
    # VLLM's deterministic DeepSeek/GLM biased top-k uses
    # torch.topk(..., sorted=False).  Megatron's local compute_topk uses the
    # default sorted=True.  The selected expert set is the same, but the
    # low-latency-compatible owner reduction consumes experts in top-k column
    # order, so the difference changes BF16 accumulation before any route
    # actually diverges.
    #
    # Only override routers registered by the DeepEP alignment bridge, and only for the
    # non-grouped Megatron path used by GLM-5 (n_group=topk_group=1 in VLLM,
    # represented as no group limit in Megatron).  Other training paths retain
    # Megatron's original semantics.
    if ORDERED_TOPK_CAPTURE_ROUTER is not None and not group_topk:
        return torch.topk(scores, k=topk, dim=1, sorted=False)

    return old_compute_topk(
        scores,
        topk,
        num_groups=num_groups,
        group_topk=group_topk,
    )


class RoutingReplay:
    all_routing_replays = []
    lazy_resources = []

    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        RoutingReplay.all_routing_replays.append(self)

    def record(self, top_indices):
        if hasattr(top_indices, "materialize_for_routing_replay"):
            self.top_indices_list.append(top_indices)
            return
        if top_indices.device.type == "cpu" and top_indices.is_pinned() and top_indices.is_contiguous():
            self.top_indices_list.append(top_indices)
            return

        # Compact non-contiguous layer views so they do not retain the full
        # all-layer routed-experts tensor once per pipeline stage.
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    @staticmethod
    def assert_all_consumed() -> None:
        errors = []
        for replay_idx, replay in enumerate(RoutingReplay.all_routing_replays):
            expected = len(replay.top_indices_list)
            if replay.forward_index != expected or replay.backward_index != expected:
                errors.append(
                    f"router={replay_idx}: forward={replay.forward_index}, "
                    f"backward={replay.backward_index}, recorded={expected}"
                )
        if errors:
            raise RuntimeError("R3 routing replay was not consumed exactly once: " + "; ".join(errors[:16]))

    def pop_forward(self):
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        if hasattr(top_indices, "materialize_for_routing_replay"):
            return top_indices.materialize_for_routing_replay("forward")
        return top_indices.to(
            accelerator.current_device(),
            dtype=torch.int32,
            non_blocking=top_indices.is_pinned(),
        )

    def pop_backward(self):
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        if hasattr(top_indices, "materialize_for_routing_replay"):
            return top_indices.materialize_for_routing_replay("backward")
        return top_indices.to(
            accelerator.current_device(),
            dtype=torch.int32,
            non_blocking=top_indices.is_pinned(),
        )

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        self.forward_index = 0

    @staticmethod
    def clear_all():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear()
        for resource in RoutingReplay.lazy_resources:
            resource.close()
        RoutingReplay.lazy_resources = []

    @staticmethod
    def clear_all_forward():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear_forward()

    @staticmethod
    def register_lazy_resource(resource):
        RoutingReplay.lazy_resources.append(resource)

    @staticmethod
    def begin_lazy_pass(release_stage):
        for resource in RoutingReplay.lazy_resources:
            resource.begin_pass(release_stage)


def get_routing_replay_compute_topk(old_compute_topk):
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            routing_replay_stage = os.environ["ROUTING_REPLAY_STAGE"]
            if routing_replay_stage == "fallthrough":
                probs, top_indices = _compute_topk_for_current_router(
                    old_compute_topk,
                    scores,
                    topk,
                    num_groups=num_groups,
                    group_topk=group_topk,
                )
            elif routing_replay_stage == "record":
                probs, top_indices = _compute_topk_for_current_router(
                    old_compute_topk,
                    scores,
                    topk,
                    num_groups=num_groups,
                    group_topk=group_topk,
                )
                ROUTING_REPLAY.record(top_indices)
            elif routing_replay_stage == "replay_forward":
                top_indices = ROUTING_REPLAY.pop_forward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"[{torch.distributed.get_rank()}] top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
            elif routing_replay_stage == "replay_backward":
                top_indices = ROUTING_REPLAY.pop_backward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                probs = scores.gather(1, top_indices)
        else:
            probs, top_indices = _compute_topk_for_current_router(
                old_compute_topk,
                scores,
                topk,
                num_groups=num_groups,
                group_topk=group_topk,
            )

        _capture_ordered_topk(top_indices)
        return probs, top_indices

    return compute_topk


def register_routing_replay(module):
    if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
        module.routing_replay = RoutingReplay()

        def pre_forward_hook(*args, **kwargs):
            set_routing_replay(module.routing_replay)

        module.register_forward_pre_hook(pre_forward_hook)


# ============================================================================
# TorchTitan backend injection
# ============================================================================


def _routing_replay_forward_hook(module, inputs, output):
    """forward_hook: replace topk_expert_ids with recorded indices in replay mode.

    Reads ROUTING_REPLAY_STAGE env var:
      - "fallthrough" / "record": no-op, return original output
      - "replay_forward": pop from forward cursor
      - "replay_backward": pop from backward cursor (AC recompute)

    torchtitan TokenChoiceTopKRouter.forward returns (topk_scores, topk_expert_ids, full scores),
    the last one carrying every expert's score with the expert dimension last ([B,L,E] here, [T,E]
    after #4121; the hook only indexes that dimension).
    The full per-expert scores come back with the output, so the replacement scores are gathered
    from them exactly the way the router itself derives topk_scores -- no gate recompute, and no
    expert_bias / node-limited logic (both only affect which experts are chosen, which is what the
    recorded ids replace). The MoE layer rebuilds its routing map and token counts from the ids
    returned here, so replacing them is enough.
    """
    stage = os.environ.get("ROUTING_REPLAY_STAGE", "fallthrough")
    if stage not in ("replay_forward", "replay_backward"):
        return  # fallthrough / record: original output

    replay = getattr(module, "routing_replay", None)
    if replay is None:
        return

    topk_scores, topk_expert_ids, full_scores = output

    if stage == "replay_forward":
        new_ids = replay.pop_forward()
    else:  # replay_backward (AC recompute)
        new_ids = replay.pop_backward()

    if new_ids.numel() != topk_expert_ids.numel():
        raise RuntimeError(
            f"routing replay: token count mismatch at layer {id(module)}, "
            f"recorded={new_ids.numel()}, expected={topk_expert_ids.numel()} "
            f"(shape {new_ids.shape} vs {topk_expert_ids.shape})"
        )

    new_ids = new_ids.reshape(topk_expert_ids.shape)

    # Gather the recorded experts' scores out of the full scores the router already returned,
    # then apply the same normalization/scaling the router applies to topk_scores.
    new_scores = full_scores.gather(dim=-1, index=new_ids)
    if getattr(module, "route_norm", False):
        denominator = new_scores.sum(dim=-1, keepdim=True) + 1e-20
        new_scores = new_scores / denominator
    new_scores = new_scores * getattr(module, "route_scale", 1.0)

    return (new_scores, new_ids, full_scores)


def apply_routing_replay_patch(model):
    """Register a RoutingReplay instance + forward_hook on each TokenChoiceTopKRouter.

    This is the torchtitan equivalent of the megatron patch
    (register_routing_replay in TopKRouter.__init__). RoutingReplay.__init__
    auto-appends to all_routing_replays. Called before parallelize_fn; parallelize
    does not change module identity, so hooks remain effective after parallelization.
    """
    from torchtitan.models.common.moe import TokenChoiceTopKRouter

    count = 0
    for module in model.modules():
        if isinstance(module, TokenChoiceTopKRouter):
            module.routing_replay = RoutingReplay()
            module.register_forward_hook(_routing_replay_forward_hook)
            count += 1
    if count == 0:
        logger.warning(
            "RoutingReplay: no TokenChoiceTopKRouter modules found in the model; "
            "routing replay hooks were not registered. Ensure the model uses "
            "torchtitan.models.common.moe.TokenChoiceTopKRouter."
        )
    else:
        logger.info("RoutingReplay: registered hooks on %d TokenChoiceTopKRouter modules", count)
    return count
