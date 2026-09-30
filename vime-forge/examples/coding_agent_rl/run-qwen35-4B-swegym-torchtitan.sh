#!/usr/bin/env bash
# Qwen3.5-4B + Codex + prepared SWE-Gym: asynchronous TorchTitan RL.
# Edit the settings and TRAIN_CMD below, or override the documented environment variables.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fail() { echo "$*" >&2; exit 2; }
DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
        cat <<'HELP'
Usage: bash examples/coding_agent_rl/run-qwen35-4B-swegym-torchtitan.sh [--dry-run]

All machine paths, AgentENV settings and training arguments are in this script.
Defaults: 2 epochs, 16 prompts x 8 attempts; task count is read from the dataset.
TP=4 / DP=1 training plus 4 TP=1 rollout engines (8 NPU dies total, default IDs 0-7).
Context and max tokens/GPU: 32768; maximum turns: 72.
Temperature 1.0, Top-P 1.0, Top-K disabled, LR 2e-6.
Async weight interval 1, TIS enabled, reward normalization disabled.
Checkpoint every 6 updates; rollout dumps on; detailed tracing off.

Example 2x2 trial (same context and turn limits; no step cap):
  SWE_ROLLOUT_BATCH_SIZE=2 SWE_N_SAMPLES_PER_PROMPT=2 SWE_SAVE_INTERVAL=off \
    bash examples/coding_agent_rl/run-qwen35-4B-swegym-torchtitan.sh

Environment overrides remain supported, including RUN_NAME, OUTPUT_DIR,
ASCEND_RT_VISIBLE_DEVICES, VIME_SLOT, SWE_ROLLOUT_BATCH_SIZE,
SWE_N_SAMPLES_PER_PROMPT and VIME_SWEGYM_PREPARED_BUNDLE.
SWE_SAVE_INTERVAL=off disables saves; SWE_DUMP_MODE=off disables rollout dumps.
SWE_LEARNING_RATE='' keeps the recipe LR. Set both VIME_AGENT_TASK_CONDA_SH
and VIME_AGENT_TASK_CONDA_PREFIX empty to use the sandbox's existing PATH.
--dry-run prints the command without activation, writes, credentials, Ray or NPU access.
If the prompt data is unavailable locally, task count and planned updates are null (unknown).
Set ADAPTER_PUBLIC_HOST to your sandbox-reachable adapter/relay host before launching;
192.0.2.1 is a documentation placeholder and is rejected outside --dry-run.
CANN defaults use /home/developer/. Override with your installed path.
Inference libraries must be compatible with the inherited worker environment.
Existing AgentENV endpoints/relay must be reachable; this script does not create them.
HELP
        exit 0 ;;
esac
(( $# == 0 )) || fail 'Use --dry-run or --help; configure this experiment with environment variables.'
source "$SCRIPT_DIR/swegym-env.sh"

# Machine environment: training Python 3.12, inference Python 3.11, CANN 9.2. Inference inherits the worker environment.
export MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/Qwen3.5-4B}"
# Training Python environment (TorchTitan and Ray).
export TRAIN_ENV="${TRAIN_ENV:-/home/developer/envs/qwen35-rl-py312}"
export VIME_TRAIN_PYTHON="${VIME_TRAIN_PYTHON:-${TRAIN_ENV}/bin/python}"
export TRAIN_PYTHON="$VIME_TRAIN_PYTHON"
# Inference Python environment: supplies the Python and vLLM executables.
export ROLLOUT_ENV="${ROLLOUT_ENV:-/home/developer/envs/qwen35-rollout-py311}"
# CANN installation providing set_env.sh.
export VIME_CANN_HOME="${VIME_CANN_HOME:-/home/developer/Ascend/cann92-isolated/cann}"
export VIME_VLLM_PYTHON="${ROLLOUT_ENV}/bin/python"
export VIME_VLLM_EXECUTABLE="${ROLLOUT_ENV}/bin/vllm"

# Prepared SWE-Gym bundle and experiment configuration.
export SWE_QWEN35_SIZE=4B SWE_TRAIN_MODE=async
export VIME_QWEN35_DRY_RUN="$DRY_RUN"
export VIME_SWEGYM_PREPARED_BUNDLE="${VIME_SWEGYM_PREPARED_BUNDLE:-${REPO_ROOT}/prepared/swegym-293}"
# Bind the prompt, templates and hidden tests to the same bundle.
export SWE_PROMPT_DATA="$VIME_SWEGYM_PREPARED_BUNDLE/rollout/train.no_gold.jsonl"
export SWE_AGENTENV_TEMPLATE_MAP="$VIME_SWEGYM_PREPARED_BUNDLE/environment/agentenv_templates.jsonl"
export SWE_VERIFIER_SPEC_MAP="$VIME_SWEGYM_PREPARED_BUNDLE/verifier/verifier_specs.jsonl"
export SWE_PREPARATION_MANIFEST="$VIME_SWEGYM_PREPARED_BUNDLE/preparation_manifest.json"
export PROMPT_DATA="$SWE_PROMPT_DATA"

# Allocation and training/rollout settings (environment overrides supported).
export SWE_TRAIN_TP="${SWE_TRAIN_TP:-4}" SWE_TRAIN_DP="${SWE_TRAIN_DP:-1}"
export SWE_ROLLOUT_GPUS="${SWE_ROLLOUT_GPUS:-4}" SWE_ROLLOUT_TP="${SWE_ROLLOUT_TP:-1}"
export SWE_ROLLOUT_BATCH_SIZE="${SWE_ROLLOUT_BATCH_SIZE:-16}"
export SWE_N_SAMPLES_PER_PROMPT="${SWE_N_SAMPLES_PER_PROMPT:-8}"
export SWE_NUM_EPOCHS="${SWE_NUM_EPOCHS:-2}"
export SWE_ROLLOUT_TEMPERATURE="${SWE_ROLLOUT_TEMPERATURE:-1.0}"
export SWE_ROLLOUT_TOP_K="${SWE_ROLLOUT_TOP_K:--1}"
export SWE_LEARNING_RATE="${SWE_LEARNING_RATE-2e-6}"
export SWE_ENABLE_PREFIX_CACHING="${SWE_ENABLE_PREFIX_CACHING:-1}"
export SWE_MAX_TOKENS_PER_GPU="${SWE_MAX_TOKENS_PER_GPU:-32768}"
export SWE_SAVE_INTERVAL="${SWE_SAVE_INTERVAL:-6}" SWE_DUMP_MODE="${SWE_DUMP_MODE:-rollout}"

# Agent sessions and verifier budgets.
export SWE_TRIAL_CONCURRENCY="${SWE_TRIAL_CONCURRENCY:-8}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-8}" SWE_BOOT_RETRIES="${SWE_BOOT_RETRIES:-3}"
export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-3600}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-3600}"
export SWE_ROLLOUT_GUARD_SEC="${SWE_ROLLOUT_GUARD_SEC:-8400}"
export SWE_VLLM_MAX_NUM_SEQS="${SWE_VLLM_MAX_NUM_SEQS:-64}"
export VIME_AGENT_MAX_TURNS_PER_SESSION="${VIME_AGENT_MAX_TURNS_PER_SESSION:-72}"
export VIME_AGENT_SANDBOX_LIFETIME_SEC="${VIME_AGENT_SANDBOX_LIFETIME_SEC:-4500}"
export VIME_AGENT_SANDBOX_RPC_RETRIES="${VIME_AGENT_SANDBOX_RPC_RETRIES:-6}"
export SWE_ROLLOUT_SEQ_LEN=32768 VIME_AGENT_MAX_CONTEXT_TOKENS=32768
export SWE_AGENT=codex SWE_TRAIN_PROTOCOL=swebench SWE_TITAN_CONFIG=vime_qwen35_4b
export VIME_AGENT_DEBUG_TRACE_DIR='' VIME_AGENT_TRACE_WRITER=''

# AgentENV connectivity, prepared runtime and guest Python.
export VIME_SWEGYM_SOURCE_PATH="${VIME_SWEGYM_SOURCE_PATH:-/home/developer/dependencies/SWE-Bench-Package-16dd480c}"
# File on the NPU host; read into E2B_API_KEY for AgentENV SDK authentication.
export VIME_AGENTENV_API_KEY_FILE="${VIME_AGENTENV_API_KEY_FILE:-/home/developer/.config/agentenv/api-key}"
# NPU -> AgentENV control/sandbox service, via an existing local SSH forward.
export E2B_API_URL="${E2B_API_URL:-http://127.0.0.1:18000}"
export E2B_SANDBOX_URL="${E2B_SANDBOX_URL:-http://127.0.0.1:18000}"
export VIME_AGENT_E2B_SDK_PATH="${VIME_AGENT_E2B_SDK_PATH-/home/developer/.venvs/agentenv-harness/lib/python3.11/site-packages}"
export VIME_AGENT_E2B_SELECTOR_MODE=template
export VIME_AGENT_TASK_CONDA_SH="${VIME_AGENT_TASK_CONDA_SH-/opt/miniconda3/etc/profile.d/conda.sh}"
export VIME_AGENT_TASK_CONDA_PREFIX="${VIME_AGENT_TASK_CONDA_PREFIX-/opt/miniconda3/envs/testbed}"
export VIME_AGENT_NODE_TARBALL="${VIME_AGENT_NODE_TARBALL:-/mnt/workspace/agentenv-artifacts/node-v22.20.0-linux-x64.tar.xz}"
export VIME_AGENT_CODEX_TARBALL="${VIME_AGENT_CODEX_TARBALL:-/mnt/workspace/agentenv-artifacts/vime-codex-offline-0.153.0.tgz}"
export VIME_AGENT_CODEX_MODEL_PROFILE=Qwen3.5-4B
unset VIME_AGENT_CODEX_EXTRA_ARGS VIME_AGENT_CODEX_EXTRA_ENVS
unset VIME_AGENT_CODEX_CONTEXT_RESERVE_TOKENS VIME_AGENT_CODEX_AUTO_COMPACT_TOKENS
unset VIME_FORK_MERGE_MAX_RESPONSE_TOKENS
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}" ADAPTER_PORT="${ADAPTER_PORT:-18001}"
# Sandbox -> model adapter/relay. Replace the documentation-only address.
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-192.0.2.1}"
export ADAPTER_PUBLIC_PORT="${ADAPTER_PUBLIC_PORT:-18080}"

# Validate settings and bundle before activating Python or creating output directories.
export VIME_SLOT="${VIME_SLOT:-31}"
validate_swegym_config
export SWE_ROLLOUTS_PER_EPOCH='' SWE_NUM_ROLLOUTS=''
if [[ -n "$SWE_PROMPT_COUNT" ]]; then
    export SWE_ROLLOUTS_PER_EPOCH=$((SWE_PROMPT_COUNT / SWE_ROLLOUT_BATCH_SIZE))
    export SWE_NUM_ROLLOUTS=$((SWE_ROLLOUTS_PER_EPOCH * SWE_NUM_EPOCHS))
fi
export RUN_NAME="${RUN_NAME:-qwen35-4B-swegym${SWE_PROMPT_COUNT:-unknown}x${SWE_N_SAMPLES_PER_PROMPT}-tp${SWE_TRAIN_TP}-dp${SWE_TRAIN_DP}-$(date +%Y%m%d-%H%M%S)-async}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/${RUN_NAME}}"
export VIME_PREPARED_ENV=1 MODE=async DET=off EVAL_ON=off SAVE_ON=off COLOCATE=off
export ASCEND_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
export VIME_RAY_PORT="${VIME_RAY_PORT:-$((6379 + VIME_SLOT))}"
export VIME_RAY_DASHBOARD_PORT="${VIME_RAY_DASHBOARD_PORT:-$((8265 + VIME_SLOT))}"
export VIME_RAY_DASHBOARD_AGENT_PORT="${VIME_RAY_DASHBOARD_AGENT_PORT:-$((52365 + VIME_SLOT))}"
export VIME_RAY_TEMP_DIR="${VIME_RAY_TEMP_DIR:-/tmp/q35-4b-s${VIME_SLOT}}"
export VIME_CKPT_ROOT="${OUTPUT_DIR}/checkpoints" VIME_TB_ROOT="${OUTPUT_DIR}/tensorboard"
export VIME_NOFILE_LIMIT="${VIME_NOFILE_LIMIT:-65535}"
SEQ_LEN="$SWE_ROLLOUT_SEQ_LEN"
MAX_RESPONSE_LEN=$((SEQ_LEN - 1))  # Adapter clamps requests to remaining context.
TRAIN_GPUS=$((SWE_TRAIN_TP * SWE_TRAIN_DP))
ROLLOUT_GPUS="$SWE_ROLLOUT_GPUS"
VLLM_RANKS_PER_ENGINE="$SWE_ROLLOUT_TP"
SERVER_CONCURRENCY=$((SWE_TRIAL_CONCURRENCY / (ROLLOUT_GPUS / VLLM_RANKS_PER_ENGINE)))

# Explicit Python entry point and every training CLI override.
# The selected recipe owns model architecture, bf16 compute, fp32 masters,
# activation checkpointing and host-backed optimizer moments.
TRAIN_CMD=(
    "$VIME_TRAIN_PYTHON" "$REPO_ROOT/train_async.py"
    --train-backend torchtitan

    # Training allocation and loss.
    --actor-num-nodes 1 --actor-num-gpus-per-node "$TRAIN_GPUS" --num-gpus-per-node "$SWE_TOTAL_NPUS"
    --train-env-vars '{"TORCH_NPU_DEVICE_CAPABILITY":"8.0"}'
    --freeze-params-name-list vision_encoder
    --micro-batch-size 1 --use-dynamic-batch-size --max-tokens-per-gpu "$SWE_MAX_TOKENS_PER_GPU"
    --balance-data --rollout-shuffle --disable-rewards-normalization
    --advantage-estimator grpo --kl-coef 0.0 --kl-loss-coef 0.0 --entropy-coef 0.0
    --eps-clip 0.2 --eps-clip-high 0.28 --calculate-per-token-loss --log-probs-chunk-size 64

    # Qwen3.5-4B TorchTitan recipe and model parallelism.
    --hf-checkpoint "$MODEL_PATH"
    # Isolate even the checkpointer's load/search path when saving is disabled.
    --checkpoint.folder "${OUTPUT_DIR}/checkpoints"
    --module vime.backends.torchtitan_utils.models.qwen3_5 --config "$SWE_TITAN_CONFIG"
    --parallelism.data_parallel_replicate_degree 1 --parallelism.data_parallel_shard_degree "$SWE_TRAIN_DP"
    --parallelism.tensor_parallel_degree "$SWE_TRAIN_TP" --parallelism.pipeline_parallel_degree 1
    --parallelism.context_parallel_degree 1 --training.seq_len "$SEQ_LEN"

    # SWE-Gym generation and sampling.
    --prompt-data "$SWE_PROMPT_DATA" --input-key prompt --label-key label --metadata-key metadata
    --rollout-function-path vime.rollout.vllm_rollout.generate_rollout
    --custom-generate-function-path examples.coding_agent_rl.generate.generate
    --rollout-batch-size "$SWE_ROLLOUT_BATCH_SIZE"
    --n-samples-per-prompt "$SWE_N_SAMPLES_PER_PROMPT" --num-steps-per-rollout 1
    --global-batch-size "$((SWE_ROLLOUT_BATCH_SIZE * SWE_N_SAMPLES_PER_PROMPT))"
    --rollout-max-prompt-len 8192 --rollout-max-response-len "$MAX_RESPONSE_LEN"
    --rollout-max-context-len "$SEQ_LEN"
    # Entry-point sampling settings; native checkpoint chat template.
    --rollout-temperature "$SWE_ROLLOUT_TEMPERATURE" --rollout-top-p 1.0 --rollout-top-k "$SWE_ROLLOUT_TOP_K"

    # vLLM inference.
    --rollout-num-gpus "$ROLLOUT_GPUS" --rollout-num-gpus-per-engine "$VLLM_RANKS_PER_ENGINE"
    # Each rollout engine uses the selected TP and internal DP=1.
    --vllm-pipeline-parallel-size 1 --vllm-data-parallel-size 1
    --vllm-max-model-len "$SEQ_LEN" --vllm-gpu-memory-utilization 0.7
    --vllm-max-num-seqs "$SWE_VLLM_MAX_NUM_SEQS" --vllm-server-concurrency "$SERVER_CONCURRENCY"
    --vllm-tool-call-parser qwen3_coder --vllm-reasoning-parser qwen3
    --vllm-compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --no-vllm-async-scheduling
    --no-vllm-weight-sync-packed
)
# Scope the LR override to the requesting launcher; other model recipes stay unchanged.
# Native AdamW reads per-group kwargs; optimizer.lr belongs to the NPU Muon schema.
if [[ -n "$SWE_LEARNING_RATE" ]]; then
    TRAIN_CMD+=(--optimizer.param-groups.0.optimizer-kwargs.lr "$SWE_LEARNING_RATE")
fi
if [[ "$SWE_ENABLE_PREFIX_CACHING" == 1 ]]; then TRAIN_CMD+=(--vllm-enable-prefix-caching); fi
# One-step-off pipelining: never pass --num-rollout alongside --num-epoch.
TRAIN_CMD+=(--num-epoch "$SWE_NUM_EPOCHS" --update-weights-interval 1 --use-tis --tis-clip 2.0 --tis-clip-low 0.0)
TRAIN_CMD+=(--use-tensorboard --tb-project-name qwen35-swegym-async --tb-experiment-name "$RUN_NAME")
# Do not apply-chat-template to dataset prompts: the Codex adapter owns templating.
# SWE_SAVE_INTERVAL=off leaves args.save_interval unset.
if [[ "$SWE_SAVE_INTERVAL" != off ]]; then
    TRAIN_CMD+=(--save "${OUTPUT_DIR}/checkpoints" --save-interval "$SWE_SAVE_INTERVAL")
    # Vime owns save cadence; TorchTitan must accept each requested save.
    TRAIN_CMD+=(--checkpoint.interval 1 --checkpoint.keep_latest_k 2)
fi
if [[ "$SWE_DUMP_MODE" != off ]]; then
    TRAIN_CMD+=(--save-debug-rollout-data "${OUTPUT_DIR}/rollout_dumps/{rollout_id}.pt")
fi

if (( DRY_RUN )); then
    print_swegym_plan
    exit 0
fi

# Preserve the verified runtime, Ray lifecycle, command quoting and log archive.
prepare_swegym_runtime
source "$REPO_ROOT/scripts/torchtitan/common/launcher.sh"
configure_run
preflight_params
configure_swegym_workers
if [[ "$SWE_DUMP_MODE" != off ]]; then mkdir -p "${OUTPUT_DIR}/rollout_dumps"; fi
cd "$REPO_ROOT"
reclaim_slot || die "could not reclaim slot ${VIME_SLOT}"
quoted_train_cmd
launch_archive "$ARCHIVE_DIR"
launch_ray_head
ray job submit \
    --address="http://127.0.0.1:${VIME_RAY_DASHBOARD_PORT}" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- "${QUOTED_TRAIN_CMD[@]}" 2>&1 | tee "${ARCHIVE_DIR}/train.log"
