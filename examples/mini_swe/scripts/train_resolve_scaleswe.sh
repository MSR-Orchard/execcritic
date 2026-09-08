#!/usr/bin/env bash
set -euo pipefail

# Full GRPO training launcher for behavior-contract behavior-contract test_patch RL on mixed patch candidates.
#
# The model learns to generate a resolve-style regression test through the behavior-contract bash-native protocol:
# inspect/edit/test -> /tmp/test_contract.json + /tmp/test_command + /tmp/test.patch -> native submit.
# Generation is Gold-free. After generation, the patch-classification reward may use labeled source
# patch candidates and Gold metadata; this reward boundary is not a Gold-hidden evaluation.
#
# Key differences from the v6 launcher:
#   * rollout    = examples.mini_swe.resolve_native_rollout.generate  (BashTestPatchResolveAgent)
#   * agent/model protocol comes from the behavior-contract RESOLVE_OVERLAY, NOT gentest.yaml
#   * generation is candidate-free AND gold-free: no starter, no test-file scaffold, no force-submit
#   * reward     = patch_classification with SWE_PATCH_CLASSIFICATION_TESTPATCH_MODE=1 (git-diff apply)
#   * Gold is read ONLY in the reward (apply gold + diff), NEVER during generation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
HARNESS_ROOT="${HARNESS_ROOT:-$REPO_ROOT/swe_harness}"
AZURE_MODAL_DIR="${AZURE_MODAL_DIR:-$HARNESS_ROOT/external/azure-modal}"
PYTHON="${PYTHON:-python3}"

# behavior-contract is fail-closed below: replacing this overlay with V10/V11 aborts before Ray starts.
RESOLVE_OVERLAY="${RESOLVE_OVERLAY:-$HARNESS_ROOT/gentest_v2/versions/behavior_contract.yaml}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"

PROJECT_NAME="${PROJECT_NAME:-code_rl_35b_resolve}"
TASK_NAME="${TASK_NAME:-qwen3.5_35B_gentest_behavior_contract}"

HF_CHECKPOINT="${HF_CHECKPOINT:-}"
REF_LOAD="${REF_LOAD:-}"
PROMPT_DATA="${PROMPT_DATA:-}"
RUN_DIR="${RUN_DIR:-}"
SAVE_DIR="${SAVE_DIR:-}"
LOAD_DIR="${LOAD_DIR:-}"
EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:-}"
REWARD_KEY="${REWARD_KEY:-raw_reward}"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-}"
LOG_DIR="${LOG_DIR:-}"
SANDBOX_PREFIX="${SANDBOX_PREFIX:-resolve-${RUN_ID//_/-}}"
AZURE_SANDBOX_MANIFEST_DIR="${AZURE_SANDBOX_MANIFEST_DIR:-}"
SANDBOX_CLEANUP_SCRIPT="$REPO_ROOT/examples/mini_swe/scripts/cleanup_azure_sandboxes.py"

WANDB_API_KEY="${WANDB_API_KEY:-}"

SANDBOX_ENV_FILE="${SANDBOX_ENV_FILE:-}"
SANDBOX_BASE_URL="${SANDBOX_BASE_URL:-}"
AZURE_CAAS_ENDPOINT="${AZURE_CAAS_ENDPOINT:-$SANDBOX_BASE_URL}"
MEGATRON_ROOT="${MEGATRON_ROOT:-}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "$name must be set explicitly; see the release ENV.example"
}

load_sandbox_api_key_file() {
  "$PYTHON" - "$1" <<'PY'
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = None
for raw_line in path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("export ") or "=" in line:
        words = shlex.split(line, comments=False, posix=True)
        if words and words[0] == "export":
            words = words[1:]
        for word in words:
            if word.startswith("SANDBOX_API_KEY="):
                value = word.split("=", 1)[1]
                break
    else:
        value = line
    if value is not None:
        break
if value is None:
    raise SystemExit(f"no SANDBOX_API_KEY value found in {path}")
value = value.strip()
value.encode("latin-1")
if "\n" in value or "\r" in value:
    raise SystemExit(f"invalid SANDBOX_API_KEY in {path}: contains newline")
print(value, end="")
PY
}

for required_name in HF_CHECKPOINT REF_LOAD PROMPT_DATA RUN_DIR SANDBOX_BASE_URL MEGATRON_ROOT; do
  require_env "$required_name"
done

SAVE_DIR="${SAVE_DIR:-$RUN_DIR/ckpt}"
LOAD_DIR="${LOAD_DIR:-$REF_LOAD}"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-$RUN_DIR/trajectories}"
LOG_DIR="${LOG_DIR:-$RUN_DIR/logs}"
AZURE_SANDBOX_MANIFEST_DIR="${AZURE_SANDBOX_MANIFEST_DIR:-$RUN_DIR/.azure_sandboxes}"

[[ -d "$REPO_ROOT" ]] || die "REPO_ROOT is not a directory: $REPO_ROOT"
[[ -d "$MEGATRON_ROOT" ]] || die "MEGATRON_ROOT is not a directory: $MEGATRON_ROOT"
[[ -f "$AZURE_MODAL_DIR/client/__init__.py" ]] || \
  die "Azure sandbox client not found under $AZURE_MODAL_DIR"
[[ -f "$SANDBOX_CLEANUP_SCRIPT" ]] || die "sandbox cleanup helper not found: $SANDBOX_CLEANUP_SCRIPT"

mkdir -p "$RUN_DIR" "$SAVE_DIR" "$TRAJECTORY_DIR" "$LOG_DIR" "$AZURE_SANDBOX_MANIFEST_DIR"
cd "$REPO_ROOT"

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_CONFIG_ENV_PREFIXES="${WANDB_CONFIG_ENV_PREFIXES:-GENTEST_,SWE_PATCH_CLASSIFICATION_,SWE_TURN_PENALTY_,RESOLVE_}"
export WANDB_CONFIG_ENV_VARS="${WANDB_CONFIG_ENV_VARS:-SWE_REWARD_MODE,SWE_EVAL_REWARD_MODE,SWE_TIMEOUT_REWARD_TOTAL,SWE_PATCH_CLASSIFICATION_NOT_SUBMITTED_SCORE,RESOLVE_OVERLAY,RESOLVE_STEP_LIMIT,RESOLVE_SHARED_VERIFY}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$MEGATRON_ROOT:$REPO_ROOT:$HARNESS_ROOT/src:$AZURE_MODAL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export SANDBOX_BASE_URL
export AZURE_CAAS_ENDPOINT
export AZURE_MODAL_PATH="$AZURE_MODAL_DIR"
export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"
export SANDBOX_PREFIX
export AZURE_SANDBOX_MANIFEST_DIR

if [[ -z "${SANDBOX_API_KEY:-}" && -n "$SANDBOX_ENV_FILE" && -f "$SANDBOX_ENV_FILE" ]]; then
  "$PYTHON" - "$SANDBOX_ENV_FILE" <<'PY'
import os
import stat
import sys

mode = stat.S_IMODE(os.stat(sys.argv[1]).st_mode)
if mode & 0o077:
    raise SystemExit(f"SANDBOX_ENV_FILE must not be group/world accessible: {sys.argv[1]} mode={mode:04o}")
PY
  SANDBOX_API_KEY="$(load_sandbox_api_key_file "$SANDBOX_ENV_FILE")"
  export SANDBOX_API_KEY
fi
if [[ -z "${SANDBOX_API_KEY:-}" ]]; then
  die "SANDBOX_API_KEY is unset; export it or set SANDBOX_ENV_FILE to a mode-0600 file"
fi

cleanup_sandbox_markers() {
  local prefix="${1:-}"
  local args=(--manifest-dir "$AZURE_SANDBOX_MANIFEST_DIR" --request-timeout "${SWE_TIMEOUT_STOP_ENV:-120}")
  if [[ -n "$prefix" ]]; then
    args+=(--prefix "$prefix")
  fi
  "$PYTHON" "$SANDBOX_CLEANUP_SCRIPT" "${args[@]}"
}

if [[ "${CLEANUP_STALE_SANDBOXES:-0}" == 1 ]]; then
  cleanup_sandbox_markers || log "WARNING: some stale sandbox markers could not be cleaned"
fi

RAY_STARTED_BY_LAUNCHER=0
RUNTIME_ENV_FILE=""

cleanup_runtime() {
  local status=$?
  trap - EXIT
  if [[ "$RAY_STARTED_BY_LAUNCHER" == 1 ]]; then
    ray stop --force >/dev/null 2>&1 || true
  fi
  cleanup_sandbox_markers "${SANDBOX_PREFIX}-" || log "WARNING: some run sandboxes remain in $AZURE_SANDBOX_MANIFEST_DIR"
  if [[ -n "$RUNTIME_ENV_FILE" && -f "$RUNTIME_ENV_FILE" ]]; then
    rm -f -- "$RUNTIME_ENV_FILE"
  fi
  exit "$status"
}

trap cleanup_runtime EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -s "$PROMPT_DATA" ]] || die "PROMPT_DATA does not exist or is empty: $PROMPT_DATA"
[[ -s "$HF_CHECKPOINT/config.json" ]] || die "HF checkpoint config is missing: $HF_CHECKPOINT/config.json"
[[ -s "$REF_LOAD/latest_checkpointed_iteration.txt" ]] || \
  die "torch-dist tracker is missing: $REF_LOAD/latest_checkpointed_iteration.txt"
[[ -z "$LOAD_DIR" || -e "$LOAD_DIR" ]] || die "LOAD_DIR does not exist: $LOAD_DIR"
[[ -s "$RESOLVE_OVERLAY" ]] || die "resolve overlay does not exist or is empty: $RESOLVE_OVERLAY"

"$PYTHON" - "$RESOLVE_OVERLAY" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
overlay = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
agent = overlay.get("agent") or {}
model = overlay.get("model") or {}
expected = {
    "agent_class": "bash_testpatch_resolve",
    "test_contract_path": "/tmp/test_contract.json",
    "require_test_contract": True,
    "require_exact_test_selector": True,
}
errors = [f"agent.{key}={agent.get(key)!r}, expected {value!r}" for key, value in expected.items() if agent.get(key) != value]
if model.get("extra_tools") != []:
    errors.append(f"model.extra_tools={model.get('extra_tools')!r}, expected []")
if errors:
    raise SystemExit("RESOLVE_OVERLAY is not the strict behavior-contract behavior contract:\n  " + "\n  ".join(errors))
PY

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [[ "$NVLINK_COUNT" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
export HAS_NVLINK
log "HAS_NVLINK=$HAS_NVLINK"

source "$REPO_ROOT/scripts/models/qwen3.5-35B-A3B.sh"

export SWE_TIMEOUT_CREATE_ENV="${SWE_TIMEOUT_CREATE_ENV:-480}"
export SWE_TIMEOUT_LLM_INFERENCE="${SWE_TIMEOUT_LLM_INFERENCE:-300}"
export SWE_TIMEOUT_GET_OBSERVATION="${SWE_TIMEOUT_GET_OBSERVATION:-120}"
export SWE_TIMEOUT_STOP_ENV="${SWE_TIMEOUT_STOP_ENV:-120}"
export SWE_TIMEOUT_REWARD_EXECUTE="${SWE_TIMEOUT_REWARD_EXECUTE:-90}"
export SWE_TIMEOUT_REWARD_TOTAL="${SWE_TIMEOUT_REWARD_TOTAL:-900}"
[[ "$SWE_TIMEOUT_REWARD_EXECUTE" =~ ^[1-9][0-9]*$ ]] || die "SWE_TIMEOUT_REWARD_EXECUTE must be positive"
[[ "$SWE_TIMEOUT_REWARD_TOTAL" =~ ^[1-9][0-9]*$ ]] || die "SWE_TIMEOUT_REWARD_TOTAL must be positive"
(( SWE_TIMEOUT_REWARD_TOTAL > SWE_TIMEOUT_REWARD_EXECUTE )) || \
  die "SWE_TIMEOUT_REWARD_TOTAL must exceed SWE_TIMEOUT_REWARD_EXECUTE"
export GENTEST_SANDBOX_TIMEOUT="${GENTEST_SANDBOX_TIMEOUT:-$SWE_TIMEOUT_CREATE_ENV}"
export SANDBOX_REQUEST_TIMEOUT="${SANDBOX_REQUEST_TIMEOUT:-$((SWE_TIMEOUT_CREATE_ENV + 60))}"
export SWE_REWARD_MODE="${SWE_REWARD_MODE:-patch_classification}"
export SWE_EVAL_REWARD_MODE="${SWE_EVAL_REWARD_MODE:-patch_classification}"
export SWE_PATCH_CLASSIFICATION_REWARD_METRIC="${SWE_PATCH_CLASSIFICATION_REWARD_METRIC:-balanced_accuracy}"
export SWE_PATCH_CLASSIFICATION_POST_REWARD_BALANCED_ACC_THRESHOLD="${SWE_PATCH_CLASSIFICATION_POST_REWARD_BALANCED_ACC_THRESHOLD:-1.0}"
export SWE_PATCH_CLASSIFICATION_POST_REWARD_SUCCESS_SCORE="${SWE_PATCH_CLASSIFICATION_POST_REWARD_SUCCESS_SCORE:-1.0}"
export SWE_PATCH_CLASSIFICATION_POST_REWARD_PARTIAL_BALANCED_ACC_THRESHOLD="${SWE_PATCH_CLASSIFICATION_POST_REWARD_PARTIAL_BALANCED_ACC_THRESHOLD:-0.8}"
export SWE_PATCH_CLASSIFICATION_POST_REWARD_PARTIAL_SCORE="${SWE_PATCH_CLASSIFICATION_POST_REWARD_PARTIAL_SCORE:-0.5}"
export SWE_PATCH_CLASSIFICATION_POST_REWARD_GOLD_SCORE="${SWE_PATCH_CLASSIFICATION_POST_REWARD_GOLD_SCORE:-0.2}"
export SWE_PATCH_CLASSIFICATION_POST_REWARD_BASE_SCORE="${SWE_PATCH_CLASSIFICATION_POST_REWARD_BASE_SCORE:-0.05}"
export SWE_PATCH_CLASSIFICATION_REQUIRE_GOLD_VALIDATED="${SWE_PATCH_CLASSIFICATION_REQUIRE_GOLD_VALIDATED:-true}"
export SWE_PATCH_CLASSIFICATION_VALIDATE_TEST="${SWE_PATCH_CLASSIFICATION_VALIDATE_TEST:-true}"
export SWE_PATCH_CLASSIFICATION_GOLD_EVAL="${SWE_PATCH_CLASSIFICATION_GOLD_EVAL:-true}"
export SWE_PATCH_CLASSIFICATION_REQUIRE_SUBMITTED="${SWE_PATCH_CLASSIFICATION_REQUIRE_SUBMITTED:-true}"
export SWE_PATCH_CLASSIFICATION_NOT_SUBMITTED_SCORE="${SWE_PATCH_CLASSIFICATION_NOT_SUBMITTED_SCORE:--0.2}"
export SWE_PATCH_CLASSIFICATION_MAX_FORMAT_ERRORS="${SWE_PATCH_CLASSIFICATION_MAX_FORMAT_ERRORS:-0}"
export SWE_PATCH_CLASSIFICATION_ENVIRONMENT_CLASS="${SWE_PATCH_CLASSIFICATION_ENVIRONMENT_CLASS:-azure_modal}"
export SWE_PATCH_CLASSIFICATION_MAX_CANDIDATES=8
export SWE_PATCH_CLASSIFICATION_INCLUDE_GOLD="${SWE_PATCH_CLASSIFICATION_INCLUDE_GOLD:-true}"
export SWE_PATCH_CLASSIFICATION_TEST_TIMEOUT="${SWE_PATCH_CLASSIFICATION_TEST_TIMEOUT:-30}"
export SWE_PATCH_CLASSIFICATION_INSTALL_TIMEOUT="${SWE_PATCH_CLASSIFICATION_INSTALL_TIMEOUT:-300}"
# Resolve reward: the generated artifact is a git-diff test_patch (may edit existing files), routed
# through run_test_patch_official (git apply) instead of the v6 single-file write.
export SWE_PATCH_CLASSIFICATION_TESTPATCH_MODE="${SWE_PATCH_CLASSIFICATION_TESTPATCH_MODE:-true}"
# Same-reward turn penalty (post-process): within a rollout group, halve raw_reward for samples
# whose reward tier is in SWE_TURN_PENALTY_TIERS and whose turn count exceeds the group's per-tier
# minimum by more than SWE_TURN_PENALTY_TURN_MARGIN. Only raw_reward is touched. Enabled by default;
# set SWE_TURN_PENALTY_ENABLE=0 to disable it explicitly.
export SWE_TURN_PENALTY_ENABLE="${SWE_TURN_PENALTY_ENABLE:-1}"
export SWE_TURN_PENALTY_TURN_MARGIN="${SWE_TURN_PENALTY_TURN_MARGIN:-8}"
export SWE_TURN_PENALTY_FACTOR="${SWE_TURN_PENALTY_FACTOR:-0.5}"
export SWE_TURN_PENALTY_TIERS="${SWE_TURN_PENALTY_TIERS:-1.0,0.5,0.2}"
export SWE_MAX_ENV_RETRIES="${SWE_MAX_ENV_RETRIES:-2}"
export SWE_ENV_RETRY_WAIT="${SWE_ENV_RETRY_WAIT:-10}"
export SWE_MAX_CREATE_ENV_RETRIES="${SWE_MAX_CREATE_ENV_RETRIES:-1}"
export SWE_CREATE_ENV_RETRY_WAIT="${SWE_CREATE_ENV_RETRY_WAIT:-10}"
export SWE_ENV_CREATE_JITTER_MAX="${SWE_ENV_CREATE_JITTER_MAX:-5}"
export SANDBOX_READY_POLL_INTERVAL="${SANDBOX_READY_POLL_INTERVAL:-20}"

export SWE_REWARD_CPU_PER_LEVEL="${SWE_REWARD_CPU_PER_LEVEL:-2}"
export SWE_REWARD_MEM_GB_PER_LEVEL="${SWE_REWARD_MEM_GB_PER_LEVEL:-8}"
export GENTEST_ENV_CPU="${GENTEST_ENV_CPU:-2}"
export GENTEST_ENV_MEMORY="${GENTEST_ENV_MEMORY:-8Gi}"

export GENTEST_TRAJECTORY_DIR="$TRAJECTORY_DIR"
# The resolve rollout loads the base gentest env/verifier config for sandbox transport only; agent
# templates come from RESOLVE_OVERLAY. No v6 starter/test-file/force-submit vars are set.
export GENTEST_CONFIG="${GENTEST_CONFIG:-$HARNESS_ROOT/src/minisweagent/config/benchmarks/gentest.yaml}"
export RESOLVE_OVERLAY
export RESOLVE_STEP_LIMIT="${RESOLVE_STEP_LIMIT:-60}"
export RESOLVE_TIME_LIMIT="${RESOLVE_TIME_LIMIT:-900}"
export RESOLVE_INSTALL_TIMEOUT="${RESOLVE_INSTALL_TIMEOUT:-1200}"
# Strict behavior-contract defaults to a separate verifier sandbox. Shared verification is an explicit throughput
# tradeoff because ignored workspace files can outlive a transactional ``git clean -fd`` reset.
export RESOLVE_SHARED_VERIFY="${RESOLVE_SHARED_VERIFY:-0}"
[[ "$RESOLVE_SHARED_VERIFY" == 0 || "$RESOLVE_SHARED_VERIFY" == 1 ]] || \
  die "RESOLVE_SHARED_VERIFY must be 0 or 1"
export GENTEST_SANITIZE_GIT_HISTORY="${GENTEST_SANITIZE_GIT_HISTORY:-1}"
export GENTEST_GIT_HISTORY_GUARD="${GENTEST_GIT_HISTORY_GUARD:-1}"
[[ "$GENTEST_SANITIZE_GIT_HISTORY" == 1 ]] || die "strict behavior-contract requires GENTEST_SANITIZE_GIT_HISTORY=1"
[[ "$GENTEST_GIT_HISTORY_GUARD" == 1 ]] || die "strict behavior-contract requires GENTEST_GIT_HISTORY_GUARD=1"
export GENTEST_CONFIG_SPECS="${GENTEST_CONFIG_SPECS:-}"
export GENTEST_CONFIG_SPECS="${GENTEST_CONFIG_SPECS:+${GENTEST_CONFIG_SPECS};;}environment.sandbox_ready_poll_interval=$SANDBOX_READY_POLL_INTERVAL;;environment.sandbox_timeout=$GENTEST_SANDBOX_TIMEOUT;;environment.request_timeout=$SANDBOX_REQUEST_TIMEOUT;;environment.cleanup_timeout=$SWE_TIMEOUT_STOP_ENV"
export SWE_PATCH_CLASSIFICATION_CONFIG_SPEC="${SWE_PATCH_CLASSIFICATION_CONFIG_SPEC:-$GENTEST_CONFIG,$HARNESS_ROOT/src/minisweagent/config/benchmarks/patch_gentest.yaml,environment.base_url=$SANDBOX_BASE_URL,environment.sandbox_ready_poll_interval=$SANDBOX_READY_POLL_INTERVAL,environment.sandbox_timeout=$SWE_TIMEOUT_CREATE_ENV,environment.request_timeout=$SANDBOX_REQUEST_TIMEOUT,environment.cleanup_timeout=$SWE_TIMEOUT_STOP_ENV}"
export GENTEST_ENVIRONMENT_CLASS="${GENTEST_ENVIRONMENT_CLASS:-azure_modal}"
export GENTEST_MAX_ALL_TOKENS=65536
export GENTEST_TIMEOUT_LLM_INFERENCE=120
export GENTEST_MAX_LLM_ATTEMPTS="${GENTEST_MAX_LLM_ATTEMPTS:-2}"
export GENTEST_MAX_ENV_RETRIES="${GENTEST_MAX_ENV_RETRIES:-1}"
export GENTEST_ENV_RETRY_WAIT="${GENTEST_ENV_RETRY_WAIT:-10}"
export GENTEST_SGLANG_TOOL_CALL_PARSER="${GENTEST_SGLANG_TOOL_CALL_PARSER:-qwen3_coder}"
export GENTEST_ENABLE_THINKING="${GENTEST_ENABLE_THINKING:-1}"
export GENTEST_UNSUBMITTED_REWARD="${GENTEST_UNSUBMITTED_REWARD:-$SWE_PATCH_CLASSIFICATION_NOT_SUBMITTED_SCORE}"
export GENTEST_DEFAULT_REWARD="${GENTEST_DEFAULT_REWARD:--1.0}"

# Isolated mode uses the canonical official Azure environment config (block_network=false), runs
# the instance's official install once, then per-test scripts skip the completed setup. Shared mode
# uses the agent workspace config and does not create this second environment.
export GENTEST_VERIFY_CONFIG="${GENTEST_VERIFY_CONFIG:-$HARNESS_ROOT/src/minisweagent/config/benchmarks/swebench_azure_modal.yaml}"
export GENTEST_VERIFY_CONFIG_SPECS="${GENTEST_VERIFY_CONFIG_SPECS:-}"
export GENTEST_VERIFY_SETUP_TIMEOUT="${GENTEST_VERIFY_SETUP_TIMEOUT:-600}"
export GENTEST_VERIFY_ASSUME_PREPARED="${GENTEST_VERIFY_ASSUME_PREPARED:-0}"
export GENTEST_VERIFY_SANDBOX_TIMEOUT="${GENTEST_VERIFY_SANDBOX_TIMEOUT:-$SWE_TIMEOUT_CREATE_ENV}"
export GENTEST_VERIFY_REQUEST_TIMEOUT="${GENTEST_VERIFY_REQUEST_TIMEOUT:-$SANDBOX_REQUEST_TIMEOUT}"
export GENTEST_VERIFY_CLEANUP_TIMEOUT="${GENTEST_VERIFY_CLEANUP_TIMEOUT:-$SWE_TIMEOUT_STOP_ENV}"
export GENTEST_VERIFY_ENV_CPU="${GENTEST_VERIFY_ENV_CPU:-2}"
export GENTEST_VERIFY_ENV_MEMORY="${GENTEST_VERIFY_ENV_MEMORY:-8Gi}"

MODEL_NAME="${MODEL_NAME:-qwen3.5-35b-a3b-resolve-testpatch-think}"
RANDOM_SUFFIX="$("$PYTHON" - <<'PY'
import random
import string

alphabet = string.ascii_lowercase + string.digits
print("".join(random.choice(alphabet) for _ in range(6)), end="")
PY
)"
export WANDB_RANDOM_SUFFIX="${WANDB_RANDOM_SUFFIX:-resolve_testpatch_${RANDOM_SUFFIX}}"
WANDB_PROJECT="${WANDB_PROJECT:-$PROJECT_NAME}"
WANDB_GROUP_NAME="${WANDB_GROUP_NAME:-$TASK_NAME}"

CKPT_ARGS=(
  --hf-checkpoint "$HF_CHECKPOINT"
  --ref-load "$REF_LOAD"
  --save "$SAVE_DIR"
  --save-interval 10
)
if [[ -n "$LOAD_DIR" ]]; then
  CKPT_ARGS+=(--load "$LOAD_DIR")
fi

CUSTOM_GENERATE_FUNCTION_PATH="${CUSTOM_GENERATE_FUNCTION_PATH:-examples.mini_swe.resolve_native_rollout.generate}"

ROLLOUT_ARGS=(
  --prompt-data "$PROMPT_DATA"
  --input-key text
  --label-key patch
  --metadata-key metadata
  --multimodal-keys '{}'
  --rollout-shuffle
  --custom-generate-function-path "$CUSTOM_GENERATE_FUNCTION_PATH"
  --custom-rollout-log-function-path examples.mini_swe.resolve_native_rollout.append_rollout_metrics
  --custom-rm-path examples.mini_swe.swe_reward.reward_func
  --custom-reward-post-process-path examples.mini_swe.post_process_rewards.patch_classification_raw_reward_normalization
  --reward-key "$REWARD_KEY"
  --num-rollout 1000
  --rollout-batch-size 16
  --n-samples-per-prompt 8
  --rollout-max-context-len 48000
  --rollout-max-response-len 8192
  --loss-mask-type "${LOSS_MASK_TYPE:-qwen3_5}"
  --rollout-temperature 1.0
  --global-batch-size 32
  --balance-data
)

EVAL_ARGS=()
if [[ "${ENABLE_EVAL:-0}" == 1 ]]; then
  [[ -s "$EVAL_PROMPT_DATA" ]] || die "EVAL_PROMPT_DATA does not exist or is empty: $EVAL_PROMPT_DATA"
  EVAL_ARGS=(
    --eval-interval 100
    --skip-eval-before-train
    --eval-prompt-data patch_cls_val "$EVAL_PROMPT_DATA"
    --eval-input-key text
    --eval-label-key patch
    --eval-reward-key "$REWARD_KEY"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 8192
    --eval-top-p 1.0
    --eval-temperature 0.0
    --eval-task-timeout 600
  )
fi

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --expert-model-parallel-size 4
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --log-probs-chunk-size 2048
   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
   --use-tis
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --ppo-epochs 1
  --use-rollout-routing-replay
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --eps-clip 0.2
  --eps-clip-high 0.28
)

GRPO_STD_NORMALIZATION="${GRPO_STD_NORMALIZATION:-0}"
case "$GRPO_STD_NORMALIZATION" in
  1) ;;
  0) GRPO_ARGS+=(--disable-grpo-std-normalization) ;;
  *) die "GRPO_STD_NORMALIZATION must be 0 or 1" ;;
esac

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

WANDB_ARGS=(
  --use-wandb
  --wandb-host "$WANDB_BASE_URL"
  --wandb-project "$WANDB_PROJECT"
  --wandb-group "$WANDB_GROUP_NAME"
  --disable-wandb-random-suffix
)
SGLANG_ARGS=(
  --sglang-moe-runner-backend triton
  --rollout-num-gpus-per-engine 1
  --sglang-server-concurrency 512
  --sglang-max-running-requests 512
  --sglang-mem-fraction-static 0.7
  --sglang-tool-call-parser qwen3_coder
  --sglang-reasoning-parser qwen3
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

export MASTER_ADDR="${MASTER_ADDR:-${VC_MASTER_HOSTS:-127.0.0.1}}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"

if ray status --address="${MASTER_ADDR}:6379" >/dev/null 2>&1; then
  die "an existing Ray cluster is reachable at ${MASTER_ADDR}:6379; refusing to take ownership"
fi
ray start --head --node-ip-address "$MASTER_ADDR" --num-gpus 8 \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
RAY_STARTED_BY_LAUNCHER=1

RUNTIME_ENV_FILE="$(mktemp "$RUN_DIR/.ray-runtime-env.XXXXXX.json")"
chmod 600 "$RUNTIME_ENV_FILE"
"$PYTHON" - "$RUNTIME_ENV_FILE" <<'PY'
import json
import os
import sys
from pathlib import Path

keys = [
    "PYTHONPATH",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE",
    "PYTORCH_CUDA_ALLOC_CONF",
    "no_proxy",
    "SANDBOX_BASE_URL",
    "SANDBOX_API_KEY",
    "SANDBOX_PREFIX",
    "AZURE_SANDBOX_MANIFEST_DIR",
    "AZURE_CAAS_ENDPOINT",
    "AZURE_MODAL_PATH",
    "MSWEA_COST_TRACKING",
    "SANDBOX_READY_POLL_INTERVAL",
    "SWE_TIMEOUT_CREATE_ENV",
    "SWE_TIMEOUT_LLM_INFERENCE",
    "SWE_TIMEOUT_GET_OBSERVATION",
    "SWE_TIMEOUT_STOP_ENV",
    "SWE_TIMEOUT_REWARD_EXECUTE",
    "SWE_TIMEOUT_REWARD_TOTAL",
    "SWE_REWARD_MODE",
    "SWE_EVAL_REWARD_MODE",
    "SWE_PATCH_CLASSIFICATION_REWARD_METRIC",
    "SWE_PATCH_CLASSIFICATION_POST_REWARD_THRESHOLD",
    "SWE_PATCH_CLASSIFICATION_POST_REWARD_BALANCED_ACC_THRESHOLD",
    "SWE_PATCH_CLASSIFICATION_POST_REWARD_PARTIAL_BALANCED_ACC_THRESHOLD",
    "SWE_PATCH_CLASSIFICATION_POST_REWARD_BASE_SCORE",
    "SWE_PATCH_CLASSIFICATION_POST_REWARD_GOLD_SCORE",
    "SWE_PATCH_CLASSIFICATION_POST_REWARD_PARTIAL_SCORE",
    "SWE_PATCH_CLASSIFICATION_POST_REWARD_SUCCESS_SCORE",
    "SWE_PATCH_CLASSIFICATION_ENVIRONMENT_CLASS",
    "SWE_PATCH_CLASSIFICATION_MAX_CANDIDATES",
    "SWE_PATCH_CLASSIFICATION_INCLUDE_GOLD",
    "SWE_PATCH_CLASSIFICATION_CONFIG_SPEC",
    "SWE_PATCH_CLASSIFICATION_TEST_TIMEOUT",
    "SWE_PATCH_CLASSIFICATION_INSTALL_TIMEOUT",
    "SWE_PATCH_CLASSIFICATION_TESTPATCH_MODE",
    "SWE_PATCH_CLASSIFICATION_VALIDATE_TEST",
    "SWE_PATCH_CLASSIFICATION_GOLD_EVAL",
    "SWE_PATCH_CLASSIFICATION_REQUIRE_GOLD_VALIDATED",
    "SWE_PATCH_CLASSIFICATION_REQUIRE_SUBMITTED",
    "SWE_PATCH_CLASSIFICATION_NOT_SUBMITTED_SCORE",
    "SWE_PATCH_CLASSIFICATION_MAX_FORMAT_ERRORS",
    "SWE_TURN_PENALTY_ENABLE",
    "SWE_TURN_PENALTY_TURN_MARGIN",
    "SWE_TURN_PENALTY_FACTOR",
    "SWE_TURN_PENALTY_TIERS",
    "SWE_MAX_ENV_RETRIES",
    "SWE_ENV_RETRY_WAIT",
    "SWE_MAX_CREATE_ENV_RETRIES",
    "SWE_CREATE_ENV_RETRY_WAIT",
    "SWE_ENV_CREATE_JITTER_MAX",
    "SWE_REWARD_CPU_PER_LEVEL",
    "SWE_REWARD_MEM_GB_PER_LEVEL",
    "SWE_REWARD_WORKERS",
    "GENTEST_ENV_CPU",
    "GENTEST_ENV_MEMORY",
    "GENTEST_NATIVE_WORKERS",
    "GENTEST_SANDBOX_TIMEOUT",
    "GENTEST_TRAJECTORY_DIR",
    "GENTEST_CONFIG",
    "GENTEST_CONFIG_SPECS",
    "GENTEST_ENVIRONMENT_CLASS",
    "GENTEST_MAX_ALL_TOKENS",
    "GENTEST_TIMEOUT_LLM_INFERENCE",
    "GENTEST_MAX_LLM_ATTEMPTS",
    "GENTEST_MAX_ENV_RETRIES",
    "GENTEST_ENV_RETRY_WAIT",
    "GENTEST_SGLANG_TOOL_CALL_PARSER",
    "GENTEST_ENABLE_THINKING",
    "GENTEST_SANITIZE_GIT_HISTORY",
    "GENTEST_GIT_HISTORY_GUARD",
    "GENTEST_VERIFY_CONFIG",
    "GENTEST_VERIFY_CONFIG_SPECS",
    "GENTEST_VERIFY_SETUP_TIMEOUT",
    "GENTEST_VERIFY_ASSUME_PREPARED",
    "GENTEST_VERIFY_SANDBOX_TIMEOUT",
    "GENTEST_VERIFY_REQUEST_TIMEOUT",
    "GENTEST_VERIFY_CLEANUP_TIMEOUT",
    "GENTEST_VERIFY_ENV_CPU",
    "GENTEST_VERIFY_ENV_MEMORY",
    "GENTEST_UNSUBMITTED_REWARD",
    "GENTEST_DEFAULT_REWARD",
    "RESOLVE_OVERLAY",
    "RESOLVE_STEP_LIMIT",
    "RESOLVE_TIME_LIMIT",
    "RESOLVE_INSTALL_TIMEOUT",
    "RESOLVE_SHARED_VERIFY",
    "WANDB_CONFIG_ENV_PREFIXES",
    "WANDB_CONFIG_ENV_VARS",
    "WANDB_RANDOM_SUFFIX",
    "WANDB_API_KEY",
    "WANDB_BASE_URL",
]
env = {key: os.environ.get(key, "") for key in keys}
env["CUDA_DEVICE_MAX_CONNECTIONS"] = env.get("CUDA_DEVICE_MAX_CONNECTIONS") or "1"
env["NCCL_NVLS_ENABLE"] = env.get("NCCL_NVLS_ENABLE") or os.environ.get("HAS_NVLINK", "0")
if not env.get("PYTORCH_CUDA_ALLOC_CONF"):
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
Path(sys.argv[1]).write_text(json.dumps({"env_vars": env}) + "\n", encoding="utf-8")
PY

log "behavior-contract behavior-contract resolve-style test_patch gentest RL"
log "run_dir=$RUN_DIR"
log "prompt_data=$PROMPT_DATA"
log "hf_checkpoint=$HF_CHECKPOINT"
log "ref_load=$REF_LOAD"
log "load_dir=$LOAD_DIR"
log "save_dir=$SAVE_DIR"
log "reward_key=$REWARD_KEY"
log "reward_post_process=examples.mini_swe.post_process_rewards.patch_classification_raw_reward_normalization"
log "grpo_std_normalization=$GRPO_STD_NORMALIZATION"
log "reward_mode=$SWE_REWARD_MODE"
log "reward_metric=$SWE_PATCH_CLASSIFICATION_REWARD_METRIC"
log "testpatch_mode=$SWE_PATCH_CLASSIFICATION_TESTPATCH_MODE"
log "require_gold_validated=$SWE_PATCH_CLASSIFICATION_REQUIRE_GOLD_VALIDATED"
log "not_submitted_score=$SWE_PATCH_CLASSIFICATION_NOT_SUBMITTED_SCORE"
log "resolve_overlay=$RESOLVE_OVERLAY"
log "resolve_step_limit=$RESOLVE_STEP_LIMIT"
log "resolve_shared_verify=$RESOLVE_SHARED_VERIFY"
log "enable_eval=${ENABLE_EVAL:-0}"
log "custom_generate_function_path=$CUSTOM_GENERATE_FUNCTION_PATH"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env="$RUNTIME_ENV_FILE" \
  -- "$PYTHON" -u train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}"

log "done: $RUN_DIR"
