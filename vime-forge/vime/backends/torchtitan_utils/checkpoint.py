# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""torchtitan checkpoint save and load, plus the RNG that torchtitan does not checkpoint.

The step directory is named by the rollout id, so the resume point is read back from the directory
name rather than stored in the checkpoint. RNG goes to per-rank sidecar files
(`step-N/rng-rank{r}.pt`): CheckpointManager saves none with an empty `states`, and per-rank
tensors would conflict inside DCP.
"""

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _rng_path(this_step: str, rank: int) -> str:
    return os.path.join(this_step, f"rng-rank{rank}.pt")


def _capture_rng_state() -> dict:
    """Capture the current rank's RNG state (cpu + device)."""
    state = {"cpu": torch.get_rng_state()}
    try:
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state()
    except Exception as e:  # noqa: BLE001 - RNG archiving is non-critical; failure must not interrupt training
        logger.warning("Failed to capture device RNG state (ignored): %s", e)
    return state


def _restore_rng_state(state: dict) -> None:
    """Restore RNG state."""
    if "cpu" in state:
        torch.set_rng_state(state["cpu"])
    try:
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda"])
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to restore device RNG state (ignored): %s", e)


def step_dir(folder: str, step: int) -> str:
    """Step directory under ``folder`` (same layout as CheckpointManager._create_checkpoint_id)."""
    return os.path.join(folder, f"step-{step}")


def save(checkpointer, args, rollout_id: int) -> None:
    """Save training checkpoint. step uses rollout_id (vime semantics), maintained by actor for alignment."""
    if getattr(args, "save", None) is None or getattr(args, "debug_rollout_only", False):
        return
    checkpointer.save(curr_step=rollout_id, last_step=False)
    dist.barrier()

    # RNG sidecar (one per rank, stored in this step's directory). Non-critical; failure only warns.
    try:
        this_step = step_dir(checkpointer.folder, rollout_id)
        if os.path.isdir(this_step):
            rank = dist.get_rank() if dist.is_initialized() else 0
            torch.save(_capture_rng_state(), _rng_path(this_step, rank))
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to save RNG sidecar (ignored): %s", e)
    dist.barrier()
    logger.info("torchtitan checkpoint saved @ rollout_id=%d -> %s", rollout_id, checkpointer.folder)


def load_from(
    checkpointer,
    checkpoint_id: str,
    *,
    model_only: bool,
    from_hf: bool = False,
    from_quantized: bool = False,
) -> None:
    """Read the checkpoint at ``checkpoint_id`` into the live components.

    The checkpointer's own folder is untouched and stays the save destination.
    """
    states = checkpointer._states_to_load(model_only)
    checkpointer.dcp_load(
        states,
        checkpoint_id=checkpoint_id,
        from_hf=from_hf,
        from_quantized=from_quantized,
    )
    logger.info(
        "Loaded %s (model_only=%s, from_hf=%s, from_quantized=%s)",
        checkpoint_id,
        model_only,
        from_hf,
        from_quantized,
    )


def load(checkpointer, args) -> int:
    """Load weights once and return the rollout id they came from, or -1 for a fresh start.

    Naming any of --load / --load-step / --load-mode reads that one checkpoint directly. Otherwise
    torchtitan decides: it seeds from initial_load_path while the checkpoint directory holds no
    step, and resumes from the largest step once it does.
    """
    weights_only = getattr(args, "load_mode", "full") == "weights"
    load_step = getattr(args, "load_step", None)

    if args.load or load_step is not None or weights_only:
        source = args.load or args.save
        step = load_step if load_step is not None else checkpointer._find_load_step(folder=source)
        assert step >= 0, f"no step-N checkpoint under {source}"
        load_from(checkpointer, step_dir(source, step), model_only=weights_only)
    else:
        step = checkpointer._find_load_step()
        checkpointer.load(step=-1)
        source = checkpointer.folder

    # Weights-only starts a new run, so the RNG stays where set_determinism left it.
    if step >= 0 and not weights_only:
        restore_rng(step_dir(source, step))
    # step-N holds the state after rollout N completed.
    return -1 if weights_only else step


def restore_rng(this_step: str) -> None:
    """Restore this rank's RNG from the sidecar in ``this_step``, if one is there."""
    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
        rng_file = _rng_path(this_step, rank)
        if os.path.isfile(rng_file):
            _restore_rng_state(torch.load(rng_file, map_location="cpu"))
            logger.info("torchtitan RNG sidecar restored (%s, rank=%d)", this_step, rank)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to restore RNG sidecar (ignored): %s", e)
