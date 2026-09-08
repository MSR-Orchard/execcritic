#!/usr/bin/env bash
# Single-node Gold/oracle-F2P-guided source-repair GRPO training launcher.
# This is not behavior-contract gentest training: the hidden F2P test is used only by the verifier/reward path.

set -euo pipefail

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
PYTHON="${PYTHON:-python3}"

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1
export MODEL_ARGS_ROTARY_BASE=10000000

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
HARNESS_ROOT="${HARNESS_ROOT:-${REPO_ROOT}/swe_harness}"
export AZURE_MODAL_PATH="${AZURE_MODAL_PATH:-${HARNESS_ROOT}/external/azure-modal}"
export SANDBOX_BASE_URL="${SANDBOX_BASE_URL:-}"
export AZURE_CAAS_ENDPOINT="${AZURE_CAAS_ENDPOINT:-${SANDBOX_BASE_URL}}"
SANDBOX_ENV_FILE="${SANDBOX_ENV_FILE:-}"
MEGATRON_ROOT="${MEGATRON_ROOT:-}"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
REF_LOAD="${REF_LOAD:-}"
PROMPT_DATA="${PROMPT_DATA:-}"
EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:-}"
RUN_DIR="${RUN_DIR:-}"
SAVE_DIR="${SAVE_DIR:-}"
export SWE_TRAJECTORY_DIR="${SWE_TRAJECTORY_DIR:-}"
AZURE_SANDBOX_MANIFEST_DIR="${AZURE_SANDBOX_MANIFEST_DIR:-}"
SANDBOX_CLEANUP_SCRIPT="${REPO_ROOT}/examples/mini_swe/scripts/cleanup_azure_sandboxes.py"

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
import os
import shlex
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
mode = stat.S_IMODE(os.stat(path).st_mode)
if mode & 0o077:
    raise SystemExit(f"SANDBOX_ENV_FILE must not be group/world accessible: {path} mode={mode:04o}")
value = None
for raw_line in path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    words = shlex.split(line, comments=False, posix=True)
    if words and words[0] == "export":
        words = words[1:]
    for word in words:
        if word.startswith("SANDBOX_API_KEY="):
            value = word.split("=", 1)[1]
            break
    if value is not None:
        break
if not value:
    raise SystemExit(f"no SANDBOX_API_KEY value found in {path}")
value.encode("latin-1")
print(value, end="")
PY
}

for required_name in HF_CHECKPOINT REF_LOAD PROMPT_DATA EVAL_PROMPT_DATA RUN_DIR SANDBOX_BASE_URL MEGATRON_ROOT; do
    require_env "$required_name"
done

SAVE_DIR="${SAVE_DIR:-${RUN_DIR}/ckpt}"
export SWE_TRAJECTORY_DIR="${SWE_TRAJECTORY_DIR:-${RUN_DIR}/trajectories}"
AZURE_SANDBOX_MANIFEST_DIR="${AZURE_SANDBOX_MANIFEST_DIR:-${RUN_DIR}/.azure_sandboxes}"
export AZURE_SANDBOX_MANIFEST_DIR

[[ -d "$MEGATRON_ROOT" ]] || die "MEGATRON_ROOT is not a directory: $MEGATRON_ROOT"
[[ -f "${AZURE_MODAL_PATH}/client/__init__.py" ]] || die "Azure sandbox client not found under ${AZURE_MODAL_PATH}"
[[ -f "$SANDBOX_CLEANUP_SCRIPT" ]] || die "sandbox cleanup helper not found: $SANDBOX_CLEANUP_SCRIPT"
if [[ -z "${SANDBOX_API_KEY:-}" && -n "$SANDBOX_ENV_FILE" && -f "$SANDBOX_ENV_FILE" ]]; then
    SANDBOX_API_KEY="$(load_sandbox_api_key_file "$SANDBOX_ENV_FILE")"
    export SANDBOX_API_KEY
fi
[[ -n "${SANDBOX_API_KEY:-}" ]] || \
    die "SANDBOX_API_KEY is unset; export it or set SANDBOX_ENV_FILE to a mode-0600 file"

export PYTHONPATH="${MEGATRON_ROOT}:${REPO_ROOT}:${HARNESS_ROOT}/src:${AZURE_MODAL_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "${REPO_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"

# SWE-specific settings (passed via environment variables)
export SWE_REWARD_MODE="repair"
export SWE_REPAIR_DIRECT_REWARD="1.5"
export SWE_REPAIR_FULL_REWARD="1.0"
export SWE_REPAIR_F2P_REWARD="0.2"
export SWE_REPAIR_SUBMIT_REWARD="0.1"
export SWE_REPAIR_ZERO_REWARD="0.0"
export SWE_CONFIG_PATH="${SWE_CONFIG_PATH:-examples/mini_swe/configs/swe_agent_v2_qwen3.5.yaml}"
export SWE_MAX_STEPS="${SWE_MAX_STEPS:-}"
export SWE_TIMEOUT_CREATE_ENV=480
export SWE_TIMEOUT_LLM_INFERENCE=60
export SWE_TIMEOUT_GET_OBSERVATION=90
export SWE_TIMEOUT_STOP_ENV=60
export SWE_TIMEOUT_REWARD_EXECUTE=90
export SWE_TIMEOUT_REWARD_TOTAL=900
export SWE_MAX_ENV_RETRIES=2
export SWE_ENV_RETRY_WAIT=10
export SWE_MAX_CREATE_ENV_RETRIES=1
export SWE_CREATE_ENV_RETRY_WAIT=10
export SWE_ENV_CREATE_JITTER_MAX=5

# F2P-guided rollout: solve normally, then keep the same sandbox/conversation
# while repairing against one deterministic gold FAIL_TO_PASS test. Each of up
# to 5 repair submissions gets its own 40-turn budget (200 repair turns max).
export SWE_F2P_REPAIR_ROUND0_STEPS=200
export SWE_F2P_REPAIR_STEPS_PER_ROUND=40
export SWE_F2P_REPAIR_MAX_ROUNDS=5
# Submit the current source diff to the normal F2P gate when a repair round
# reaches its turn limit. Set to 0 to retain the legacy LimitsExceeded behavior.
export SWE_F2P_REPAIR_FORCE_SUBMIT_ON_TURN_LIMIT="${SWE_F2P_REPAIR_FORCE_SUBMIT_ON_TURN_LIMIT:-1}"
export SWE_F2P_REPAIR_N_TESTS=1
export SWE_F2P_REPAIR_TEST_TIMEOUT=240
export SWE_F2P_REPAIR_SETUP_TIMEOUT=600
export SWE_F2P_REPAIR_FEEDBACK_MAX_CHARS=6000

# Key hyperparameters (used in argument arrays and wandb naming)
MODEL_NAME="qwen3.5-35b-a3b"
OVER_SAMPLING_BATCH_SIZE=8
ROLLOUT_BATCH_SIZE=8
N_SAMPLES=8
GLOBAL_BATCH_SIZE=64    # = ROLLOUT_BATCH_SIZE * N_SAMPLES
ADVANTAGE_ESTIMATOR="grpo"
LR="1e-6"
NUM_NODES=1
PPO_EPOCHS=1

RANDOM_SUFFIX="$("$PYTHON" - <<'PY'
import secrets
import string

alphabet = string.ascii_lowercase + string.digits
print("".join(secrets.choice(alphabet) for _ in range(6)), end="")
PY
)"
export WANDB_RANDOM_SUFFIX="qwen3.5-35b-a3b_grpo_${RANDOM_SUFFIX}"
WANDB_GROUP_NAME="p1044-d0514v2-pe${PPO_EPOCHS}_r${ROLLOUT_BATCH_SIZE}_n${N_SAMPLES}_g${GLOBAL_BATCH_SIZE}_${ADVANTAGE_ESTIMATOR}_lr${LR}_${NUM_NODES}nodes_${WANDB_RANDOM_SUFFIX}"

if [[ ! -s "${HF_CHECKPOINT}/config.json" ]]; then
    echo "Missing HF checkpoint config: ${HF_CHECKPOINT}/config.json" >&2
    exit 1
fi
if [[ ! -s "${REF_LOAD}/latest_checkpointed_iteration.txt" ]]; then
    echo "Missing torch_dist checkpoint tracker: ${REF_LOAD}/latest_checkpointed_iteration.txt" >&2
    exit 1
fi
REF_LOAD_ITERATION="$(tr -d '[:space:]' < "${REF_LOAD}/latest_checkpointed_iteration.txt")"
if [[ "${REF_LOAD_ITERATION}" == "release" ]]; then
    REF_LOAD_CHECKPOINT_DIR="${REF_LOAD}/release"
elif [[ "${REF_LOAD_ITERATION}" =~ ^[0-9]+$ ]]; then
    printf -v REF_LOAD_CHECKPOINT_DIR '%s/iter_%07d' "${REF_LOAD}" "$((10#${REF_LOAD_ITERATION}))"
else
    echo "Invalid torch_dist checkpoint tracker value: ${REF_LOAD_ITERATION}" >&2
    exit 1
fi
if [[ ! -s "${REF_LOAD_CHECKPOINT_DIR}/common.pt" ]]; then
    echo "Missing torch_dist common state: ${REF_LOAD_CHECKPOINT_DIR}/common.pt" >&2
    exit 1
fi
if ! find "${REF_LOAD_CHECKPOINT_DIR}" -type f -name '*.distcp' -size +0c -print -quit | grep -q .; then
    echo "No non-empty torch_dist shards under ${REF_LOAD_CHECKPOINT_DIR}" >&2
    exit 1
fi
if [[ ! -s "${PROMPT_DATA}" ]]; then
    echo "Missing or empty training prompt data: ${PROMPT_DATA}" >&2
    exit 1
fi
if [[ ! -s "${EVAL_PROMPT_DATA}" ]]; then
    echo "Missing or empty evaluation prompt data: ${EVAL_PROMPT_DATA}" >&2
    exit 1
fi
if [[ ! -s "${SWE_CONFIG_PATH}" ]]; then
    echo "Missing or empty SWE config: ${SWE_CONFIG_PATH}" >&2
    exit 1
fi

"$PYTHON" - "$PROMPT_DATA" "$EVAL_PROMPT_DATA" <<'PY'
import json
import re
import sys
from pathlib import Path

safe_id = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
for filename in sys.argv[1:]:
    path = Path(filename)
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or not str(row.get("problem_statement") or "").strip():
                raise SystemExit(f"{path}:{line_number}: nonempty problem_statement is required")
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                raise SystemExit(f"{path}:{line_number}: metadata must be an object")
            instance_id = str(metadata.get("instance_id") or "")
            if not safe_id.fullmatch(instance_id):
                raise SystemExit(f"{path}:{line_number}: unsafe metadata.instance_id={instance_id!r}")
            if not any(metadata.get(key) for key in ("test_patch", "f2p_patch", "f2p_script")):
                raise SystemExit(
                    f"{path}:{line_number}: metadata needs test_patch, f2p_patch, or f2p_script"
                )
            if not metadata.get("FAIL_TO_PASS") and not metadata.get("f2p_script"):
                raise SystemExit(f"{path}:{line_number}: metadata needs FAIL_TO_PASS or f2p_script")
    if rows == 0:
        raise SystemExit(f"{path}: no JSONL rows")
    print(f"validated {rows} oracle-F2P rows: {path}")
PY

mkdir -p "${RUN_DIR}" "${SAVE_DIR}" "${SWE_TRAJECTORY_DIR}" "${AZURE_SANDBOX_MANIFEST_DIR}"

CKPT_ARGS=(
   # SGLang/tokenizer use HF assets; Megatron initializes from the converted weights.
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_LOAD}"
   --load "${SAVE_DIR}"
   --save "${SAVE_DIR}"

   --save-interval 10
   #--save-interval 1

)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key problem_statement
   # The top-level Gold source patch is intentionally not loaded into Sample.label. Oracle F2P
   # content comes from metadata.test_patch/f2p_patch/f2p_script and remains verifier-only.
   #--apply-chat-template
   # qwen3.5 ships a multimodal processor, so prompts must be in conversation (list) form.
   # An empty multimodal map forces list form WITHOUT applying a chat template (avoids double-templating).
   --multimodal-keys '{}'
   --rollout-shuffle

   --custom-generate-function-path examples.mini_swe.f2p_repair_rollout.generate
   --custom-rollout-log-function-path examples.mini_swe.f2p_repair_rollout.append_rollout_metrics
   --custom-rm-path examples.mini_swe.repair_reward.reward_func
   --reward-key raw_reward
   # Mean-center raw 0/0.1/0.2/1/1.5 rewards by group_index; do not divide by group std.
   --custom-reward-post-process-path examples.mini_swe.repair_reward.center_rewards_by_group
   # Keep every GRPO group, including zero-std groups. Aborted rollout slots stay
   # in the group but have zero training loss, so no group is dropped or refilled.
   --dynamic-sampling-filter-path examples.mini_swe.gentest_filters.mask_aborted
   --calculate-per-token-loss

   --num-rollout 300
   #--over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES}
   --rollout-max-response-len 6122
   --rollout-temperature 1.0
   --rollout-top-p 1.0

   # # qwen3.5 stop token ids (im_end / endoftext in the 248320 vocab)
   # --rollout-stop-token-ids 248046 248044

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 3000
   --skip-eval-before-train
   --eval-prompt-data swe_val "${EVAL_PROMPT_DATA}"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 8192
   --eval-top-p 0.95
   --eval-temperature 0.95
   --eval-task-timeout 1500
)

PERF_ARGS=(
   # Single-node actor parallelism: TP2 x CP4 x DP1 = 8 GPUs.
   # Actor/training TP. Independent of SGLang serving TP (the GDN weight-sync bug was
   # on the serving side, fixed via DP-attention in SGLANG_ARGS). TP=1 OOMs the 35B
   # training model, so keep TP=2 and use the single-node EP=4 layout.
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   # This language-only HF export has no mtp.* weights. Do not instantiate a
   # randomly initialized MTP block from the nested Qwen3.5 config.
   --mtp-num-layers 0

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator ${ADVANTAGE_ESTIMATOR}
   # PPO epochs: train each rollout batch this many times (off-policy passes
   # corrected by eps-clip). 1 = on-policy single pass. LR schedule still
   # advances once per rollout, not per epoch.
   --ppo-epochs ${PPO_EPOCHS}
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   # --entropy-coef 0.00
   # --entropy-coef 1e-3
   # --entropy-coef 1e-4
   --eps-clip 0.2
   --eps-clip-high 0.28
   # --normalize-advantages
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR}
   --lr-decay-style cosine
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=()
if [[ "${ENABLE_WANDB:-1}" == 1 ]]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-host "${WANDB_BASE_URL}"
      --wandb-project "${WANDB_PROJECT:-slime-swe-agent}"
      --wandb-group "${WANDB_GROUP_NAME}"
      --disable-wandb-random-suffix
   )
elif [[ "${ENABLE_WANDB:-1}" != 0 ]]; then
   die "ENABLE_WANDB must be 0 or 1"
fi



SGLANG_ARGS=(
   --sglang-moe-runner-backend triton
   #--sglang-attention-backend trtllm_mha
   # Engine spans 2 GPUs. We use DP-attention (+EP) instead of TP for serving so the
   # hybrid GDN params (conv1d / A_log / dt_bias / in_proj_*) are REPLICATED, not
   # TP-sharded. SGLang 0.5.12.post1's update_weights_from_tensor mis-shards those GDN
   # params under engine-TP>1 -> garbage rollout right after the first weight sync.
   # DP-attention sidesteps that buggy path; experts are EP-sharded for memory.
   --rollout-num-gpus-per-engine 2
   --sglang-enable-dp-attention
   --sglang-data-parallel-size 2
   --sglang-expert-parallel-size 2
   --sglang-server-concurrency 64
   --sglang-max-running-requests 64
   #--sglang-chunked-prefill-size 8192
   --sglang-chunked-prefill-size 16384
   --sglang-mem-fraction-static 0.7

   # --sglang-disable-radix-cache
   # --sglang-mamba-scheduler-strategy no_buffer

   --sglang-mamba-scheduler-strategy extra_buffer

   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash

   # qwen3.5 specific: flex MoE dispatcher + DeepEP (overrides alltoall from model config)
   --moe-token-dispatcher-type flex
   --moe-enable-deepep
)



# === Single-node Ray setup ===
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"

RAY_STARTED_BY_LAUNCHER=0
RUNTIME_ENV_FILE=""

cleanup_runtime() {
   local status=$?
   trap - EXIT
   if [[ "$RAY_STARTED_BY_LAUNCHER" == 1 ]]; then
      ray stop --force >/dev/null 2>&1 || true
   fi
   "$PYTHON" "$SANDBOX_CLEANUP_SCRIPT" \
      --manifest-dir "$AZURE_SANDBOX_MANIFEST_DIR" \
      --request-timeout "$SWE_TIMEOUT_STOP_ENV" || \
      log "WARNING: some run-owned sandbox markers remain in $AZURE_SANDBOX_MANIFEST_DIR"
   if [[ -n "$RUNTIME_ENV_FILE" && -f "$RUNTIME_ENV_FILE" ]]; then
      rm -f -- "$RUNTIME_ENV_FILE"
   fi
   exit "$status"
}

trap cleanup_runtime EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# The local head is the only Ray node.
if ray status --address="${MASTER_ADDR}:6379" >/dev/null 2>&1; then
   die "an existing Ray cluster is reachable at ${MASTER_ADDR}:6379; refusing to take ownership"
fi
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
RAY_STARTED_BY_LAUNCHER=1

# Keep credentials out of argv: Ray reads them from a mode-0600 runtime-env file.
export SANDBOX_API_KEY WANDB_API_KEY
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
RUNTIME_ENV_FILE="$(mktemp "$RUN_DIR/.ray-runtime-env.XXXXXX.json")"
chmod 600 "$RUNTIME_ENV_FILE"
"$PYTHON" - "$RUNTIME_ENV_FILE" <<'PY'
import json
import os
import sys
from pathlib import Path

keys = [
    "PYTHONPATH",
    "SANDBOX_BASE_URL",
    "SANDBOX_API_KEY",
    "AZURE_CAAS_ENDPOINT",
    "AZURE_MODAL_PATH",
    "AZURE_SANDBOX_MANIFEST_DIR",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE",
    "no_proxy",
    "SWE_REWARD_MODE",
    "SWE_REPAIR_DIRECT_REWARD",
    "SWE_REPAIR_FULL_REWARD",
    "SWE_REPAIR_F2P_REWARD",
    "SWE_REPAIR_SUBMIT_REWARD",
    "SWE_REPAIR_ZERO_REWARD",
    "SWE_MAX_STEPS",
    "SWE_CONFIG_PATH",
    "SWE_TIMEOUT_CREATE_ENV",
    "SWE_TIMEOUT_LLM_INFERENCE",
    "SWE_TIMEOUT_GET_OBSERVATION",
    "SWE_TIMEOUT_STOP_ENV",
    "SWE_TIMEOUT_REWARD_EXECUTE",
    "SWE_TIMEOUT_REWARD_TOTAL",
    "SWE_MAX_ENV_RETRIES",
    "SWE_ENV_RETRY_WAIT",
    "SWE_MAX_CREATE_ENV_RETRIES",
    "SWE_CREATE_ENV_RETRY_WAIT",
    "SWE_ENV_CREATE_JITTER_MAX",
    "SWE_F2P_REPAIR_ROUND0_STEPS",
    "SWE_F2P_REPAIR_STEPS_PER_ROUND",
    "SWE_F2P_REPAIR_MAX_ROUNDS",
    "SWE_F2P_REPAIR_FORCE_SUBMIT_ON_TURN_LIMIT",
    "SWE_F2P_REPAIR_N_TESTS",
    "SWE_F2P_REPAIR_TEST_TIMEOUT",
    "SWE_F2P_REPAIR_SETUP_TIMEOUT",
    "SWE_F2P_REPAIR_FEEDBACK_MAX_CHARS",
    "WANDB_API_KEY",
    "WANDB_BASE_URL",
]
Path(sys.argv[1]).write_text(
    json.dumps({"env_vars": {key: os.environ.get(key, "") for key in keys}}) + "\n",
    encoding="utf-8",
)
PY

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env="${RUNTIME_ENV_FILE}" \
   -- "$PYTHON" -u train.py \
   --actor-num-nodes "${NUM_NODES}" \
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

log "training complete: $RUN_DIR"
