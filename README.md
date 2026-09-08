# behavior-contract generated-test, self-repair, and SWE-RL training kit

This kit captures the code path used to generate resolve-style behavior-contract regression tests, turn those
tests into a canonical map, repair source patches behind an isolated generated-test gate, and run
official SWE-bench verification. It also includes the slime runtime and the complete first-party
dependency closure for these two requested Qwen3.5-35B-A3B training entrypoints:

- `examples/mini_swe/scripts/train_swe_qwen3.5_35B_A3B_repair.sh`
- `examples/mini_swe/scripts/train_resolve_scaleswe.sh`

The training launchers are related research paths, not stages of the strict behavior-contract evaluation
pipeline. Their protocol and reward differences are called out below.

behavior-contract is a **behavior-contract protocol**, not a model or checkpoint. During generation, the model
sees the issue, buggy checkout, public documentation, and current tests; it does not see the Gold
source patch or oracle test. The optional Gold check in `batch_resolve.py` happens only after the
model has submitted and is an offline audit.

```text
issue + buggy Base
        |
        v
behavior-contract generation -- exact test nodes + behavior contract
        |
        v
results.jsonl -- build_testpatch_testmap.py --> {id: [{test_patch, test_command}]}
                                                    |
Round-0 source patch -------------------------------+
                                                    v
                                  isolated generated-test repair gate
                                                    |
                                                    v
                                  final patch + official verification

training branch A: issue -> Round-0 source patch -> Gold F2P-guided repair -> GRPO reward
training branch B: issue -> resolve-style test_patch -> candidate-patch classification -> GRPO reward
```

## Included code

| Path | Role |
|---|---|
| `swe_harness/gentest_v2/versions/behavior_contract.yaml` | Model-visible behavior-contract contract and prompt |
| `swe_harness/gentest_v2/batch_resolve.py` | Parallel generation, Base gate, optional post-submit Gold audit |
| `swe_harness/src/minisweagent/agents/extra/bash_testpatch_resolve.py` | Native bash submission and behavior-contract contract validation |
| `swe_harness/scripts/repair/build_testpatch_testmap.py` | behavior-contract results to canonical repair-map exporter |
| `swe_harness/scripts/repair/oracle_f2p_full_repair_verify_loop.py` | Canonical Round-0 + repair + official-verify orchestrator; use `--gentest-map` |
| `swe_harness/scripts/repair/cache/selfrepair_gentest.py` | Generated-test gate and repair lifecycle |
| `swe_harness/src/minisweagent/` | Matching mini-swe-agent runtime, configs, and official verifiers |
| `swe_harness/external/azure-modal/client/` | Sandbox client used by the harness and training paths |
| `train.py`, `slime/`, `slime_plugins/` | Matching slime training runtime and Qwen3.5 model plugin |
| `scripts/models/qwen3.5-35B-A3B.sh` | Megatron model arguments shared by both launchers |
| `examples/mini_swe/f2p_repair_rollout.py` | Source-patch Round-0 and F2P-guided repair rollout |
| `examples/mini_swe/repair_reward.py` | Source-repair raw reward and group centering |
| `examples/mini_swe/resolve_native_rollout.py` | Native resolve-style test-patch rollout |
| `examples/mini_swe/swe_reward.py` | Patch-classification reward implementation |
| `examples/mini_swe/post_process_rewards.py` | Patch-classification reward normalization and turn penalty |

Results, trajectories, checkpoints, datasets, credentials, proxy logs, and sandbox state are
deliberately excluded.

## What you supply

1. Python 3.11 or newer and access to the Hugging Face benchmark dataset/cache.
2. An Azure-modal-compatible sandbox pool via `SANDBOX_BASE_URL` and `SANDBOX_API_KEY`.
3. A ready model endpoint and its credentials. Check `/v1/models` and one real completion before a
   run. For `trapi_response`, also provide `TRAPI_BEARER_TOKEN_FILE`, `TRAPI_BEARER_TOKEN`, or
   `TRAPI_ALLOW_AZ_CRED=1`.
4. For provided-patch repair, Round-0 mini-swe-agent trajectory JSONL with instance IDs and source
   patches. The orchestrator can instead generate Round 0 with `--fresh-solve-round0`.
5. For training, a compatible CUDA image with eight visible GPUs, Ray, Megatron-LM, SGLang,
   Transformer Engine, DeepEP, FlashAttention/flash-linear-attention, and the Qwen3.5
   tokenizer/model runtime. Megatron-LM and native CUDA dependencies are not vendored.
6. The exact HF and torch-distributed checkpoints plus launcher-specific prompt/eval JSONL. Models,
   checkpoints, datasets, W&B state, and sandbox credentials are not in the archive.

The builder excludes credential-like filenames and scans selected files for common private-key and
W&B-key markers; those checks found no match in this snapshot. They are pattern-based checks, not a
guarantee that arbitrary secret formats are absent, so run an independent release scan before
redistribution.

## Install the harness tools

Run from the extracted bundle root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-harness.txt

export PYTHONPATH="$PWD/swe_harness/src:$PWD/swe_harness/external/azure-modal${PYTHONPATH:+:$PYTHONPATH}"
export SANDBOX_BASE_URL="https://your-sandbox-endpoint"
export SANDBOX_API_KEY="..."
```

The bundle pins `swebench==4.1.0`, matching the environment in which this snapshot was validated.
The project itself is installed editable from the bundled `swe_harness/pyproject.toml`.

For GPU training, start from a known-compatible slime image instead of treating the harness venv as
a complete training environment. The bundle preserves slime's root `requirements.txt`,
`pyproject.toml`, and `setup.py`, but they do not lock CUDA, Megatron-LM, SGLang, or kernel builds.
Inside that image:

```bash
python -m pip install -e . --no-deps
python -m pip install -e ./swe_harness[azure_modal] --no-deps
export MEGATRON_ROOT=/absolute/path/to/Megatron-LM
```

## 1. Generate behavior-contract tests

Start with one instance and one worker. The command below assumes a local OpenAI-compatible model
server on port 8000; replace the model path and port with the endpoint you actually verified.

```bash
GEN_OUT=/data/runs/behavior_contract_smoke

python -u swe_harness/gentest_v2/batch_resolve.py \
  --output "$GEN_OUT" \
  --subset verified --split test \
  --model hosted_vllm//absolute/path/to/model \
  --model-class litellm_response \
  --host 127.0.0.1 --ports 8000 \
  --workers 1 --limit 1 \
  --step-limit 60 --instance-timeout 900 --install-timeout 1200 \
  --overlay swe_harness/gentest_v2/versions/behavior_contract.yaml \
  --tool-choice auto --reasoning-effort medium \
  --max-output-tokens 8192 \
  --no-gold-check
```

Important defaults and switches:

- Verification is isolated unless `--shared-verify` is explicitly supplied.
- Git-history sanitization and the runtime git-history guard default on. Keep them on for a strict
  Gold-hidden run.
- Omit `--no-gold-check` only when you intentionally want the post-submission Gold audit. Gold
  fields are never used by the map exporter for selection.
- Use `--tool-choice auto` for Qwen/SGLang. Backend-specific tool-choice behavior should be smoked
  before scaling out.
- A process or allocated GPU is not success. Require a persisted row in `results.jsonl`, a saved
  trajectory, `submitted=true`, nonempty `test_patch`/`test_command`, and `base_clean_fail=true`.

For a full run, remove `--limit 1`, set the exact ID file with `--ids`, and increase workers only
after the smoke persists correctly. Never resume shared verification into an isolated-verification
result file, or vice versa.

## 2. Export the canonical behavior-contract map

behavior-contract uses a test patch plus its exact command. Do **not** use the legacy
`build_gentest_testmap.py`, which exports full-file `test_code` records.

```bash
TEST_MAP="$GEN_OUT/behavior_contract.testmap.max1.json"

python swe_harness/scripts/repair/build_testpatch_testmap.py \
  --results "$GEN_OUT/results.jsonl" \
  --out "$TEST_MAP" \
  --max-tests-per-instance 1
```

The output is:

```json
{
  "org__repo-123": [
    {
      "test_patch": "diff --git a/tests/test_x.py b/tests/test_x.py\n...",
      "test_command": "python -m pytest -q tests/test_x.py::test_issue_behavior",
      "source_trajectory": "/path/to/trajectory.json",
      "source_exit_status": "Submitted"
    }
  ]
}
```

The adjacent `*.summary.json` reports Base/Gold counts for audit, but the map contains no Gold
labels and selection is only submitted + nonempty canonical patch + exact command.

## 3. Run self-repair

The canonical entrypoint is the orchestrator below. This example consumes existing Round-0
trajectories. It keeps the generated-test gate in a separate reset-before-grade sandbox.

```bash
SOURCE_JSONL=/data/runs/round0/trajectories.jsonl
REPAIR_OUT=/data/runs/behavior_contract_selfrepair

python -u swe_harness/scripts/repair/oracle_f2p_full_repair_verify_loop.py \
  --harness-root swe_harness \
  --input "$SOURCE_JSONL" \
  --output "$REPAIR_OUT" \
  --gentest-map "$TEST_MAP" \
  --subset verified --split test \
  --endpoint "$SANDBOX_BASE_URL" \
  --model hosted_vllm//absolute/path/to/model \
  --model-class litellm \
  --api-base http://127.0.0.1:8000/v1 --api-key EMPTY \
  --workers 1 --rollouts-per-source 1 \
  --max-rounds 5 --max-gate-checks 5 --max-turns 30 \
  --max-tests 1 \
  --isolated-generated-test-gate
```

For a fresh Round-0 solve selected directly from the dataset, add
`--fresh-solve-round0 --dataset-instances --round0-turns 200`. The parser still requires `--input`,
but its contents are not consumed when `--dataset-instances` supplies the selected instances.

For strict behavior-contract, intentionally **omit** `--apply-test-patch-at-start`: enabling it applies the
dataset's official test patch before the generated test and can leak or conflict with the behavior-contract
gate. Keep `--isolated-generated-test-gate` enabled. The map preserves the submitted
`test_command`, while the self-repair executor removes the known-incompatible pytest
`--no-header` flag when present and wraps execution with testbed activation plus a return-code
marker.

Primary outputs under `$REPAIR_OUT` are:

- `run_config.json`: effective model, dataset, repair, gate, and isolation settings;
- `source_manifest.jsonl` and `.summary.json`: eligible Round-0 units;
- `completed.jsonl`: one durable result per attempt, including Round-0/final patches, gate status,
  repair turns, and official resolution;
- per-attempt repair JSON and trajectories under the row directories.

Resume with the same output directory only when the input, map, model, and protocol are unchanged.
The completed-attempt keys are used for skipping; use a new output directory for a changed setting.

## 4. Independent full verification

A generated-test gate pass or “rescue” is not an official SWE-bench resolution. The orchestrator
runs an official verifier for each selected final patch, but publication-quality comparisons should
also compose the full denominator and verify it separately.

```bash
FINAL_INPUT="$REPAIR_OUT/final_verify_input"

python swe_harness/scripts/repair/build_full_verify_input_from_repair.py \
  --source-run /data/runs/round0/trajectory_directory \
  --repair-jsonl "$REPAIR_OUT/completed.jsonl" \
  --out "$FINAL_INPUT"

python -m minisweagent.run.utilities.mini_extra \
  swebench-verify-azure-modal-cli \
  -c swebench \
  --subset verified --split test \
  --output "$FINAL_INPUT" \
  --workers 16
```

Report official outcomes joined by `instance_id` and retain the full denominator. Keep generated-
test gate transitions, official patch outcomes, and infrastructure errors as separate buckets.

## 5. Train the source-patch self-repair policy

`train_swe_qwen3.5_35B_A3B_repair.sh` trains a source-code repair policy. Each sample first performs
a normal Round-0 solve, then may continue in the same conversation/workspace against one
deterministic Gold `FAIL_TO_PASS` test. This is an oracle-guided training signal; it does not consume
the behavior-contract generated-test map and must not be reported as strict Gold-hidden behavior-contract evaluation.

The requested launcher uses 8 prompt groups x 8 samples = 64 rollout slots. It keeps zero-standard-
deviation groups, masks aborted samples from training loss, and mean-centers raw reward over the
non-aborted samples without dividing by group standard deviation. Raw reward tiers are `1.5`
(Round-0 official resolve), `1.0` (official resolve after repair), `0.2` (selected F2P pass without
official resolution), `0.1` (valid unresolved submission), and `0.0` (no valid submission or an
integrity failure).

Override all machine-specific inputs explicitly:

```bash
export MEGATRON_ROOT=/opt/Megatron-LM
export HF_CHECKPOINT=/data/checkpoints/iter_0000019_hf
export REF_LOAD=/data/checkpoints/iter_0000019_torch_dist
export PROMPT_DATA=/data/prompts/source_repair_train.jsonl
export EVAL_PROMPT_DATA=/data/prompts/swebench_eval.jsonl
export RUN_DIR=/data/runs/qwen35b_source_repair
export SANDBOX_BASE_URL=https://your-sandbox-endpoint
export SANDBOX_API_KEY=...
export WANDB_API_KEY=...  # optional if W&B is already authenticated

CLEANUP_PROCESSES=0 \
bash examples/mini_swe/scripts/train_swe_qwen3.5_35B_A3B_repair.sh
```

The launcher checks that the prompt/eval/config files, HF config, and tracked torch-distributed
checkpoint exist and are nonempty, then writes checkpoints to `$RUN_DIR/ckpt` and rollout/reward
artifacts to `$RUN_DIR/trajectories`. It does not schema-validate the JSONL. A source-repair row
needs top-level `problem_statement`, `patch`, and `metadata`; the metadata must supply the verifier
identity/repository fields (`instance_id`, `repo`, and `base_commit`), `FAIL_TO_PASS` and
`PASS_TO_PASS`, at least one of `test_patch`/`f2p_patch`/`f2p_script`, and any dataset-specific
image/install fields used by the official verifier.

Startup process cleanup defaults off. On a dedicated node, explicitly setting
`CLEANUP_PROCESSES=1` force-kills local `sglang`, Ray, and Python processes before starting a new
eight-GPU Ray head. Even with cleanup disabled, the launcher starts its own Ray head; independently
ensure that port 8265 and the GPUs are free.

## 6. Train the resolve-style test-patch policy

`train_resolve_scaleswe.sh` trains generation of a resolve-style regression `test_patch`. The model
gets the issue and repository but no candidate or Gold patch during generation. The reward later
applies the submitted test patch to positive/negative source-patch candidates and computes patch-
classification reward; Gold is therefore reward-side supervision, not model-visible generation
context.

The launcher defaults to the V11 native-bash overlay, 16 prompt groups, 8 samples per prompt,
`global-batch-size=32`, no GRPO standard-deviation normalization, shared workspace verification, and
a same-tier turn penalty. Although `resolve_native_rollout.py` can parse the behavior-contract overlay, changing
`RESOLVE_OVERLAY` alone does not establish that the dataset and patch-classification reward satisfy
the strict behavior-contract behavior contract. Re-run the contract and leakage checks before calling such a run
behavior-contract training.

```bash
export MEGATRON_ROOT=/opt/Megatron-LM
export HF_CHECKPOINT=/data/checkpoints/resolve_iter29_hf
export REF_LOAD=/data/checkpoints/resolve_iter29_torch_dist
export LOAD_DIR="$REF_LOAD"
export PROMPT_DATA=/data/prompts/scaleswe_rebench_4pos4neg.jsonl
export RUN_DIR=/data/runs/qwen35b_resolve_scaleswe
export SANDBOX_BASE_URL=https://your-sandbox-endpoint
export SANDBOX_API_KEY=...
export WANDB_API_KEY=...  # optional if W&B is already authenticated

CLEANUP_PROCESSES=0 CLEANUP_STALE_SANDBOXES=0 \
bash examples/mini_swe/scripts/train_resolve_scaleswe.sh
```

Training rows must provide `text`, `patch`, and `metadata`; the reward metadata must carry the
positive/negative patch candidates expected by `swe_reward.py`. Set `ENABLE_EVAL=1` and
`EVAL_PROMPT_DATA=/path/to/val.jsonl` to enable periodic evaluation. Outputs are `$RUN_DIR/ckpt`,
`$RUN_DIR/trajectories`, `$RUN_DIR/logs`, and run-owned sandbox marker files under
`$RUN_DIR/.azure_sandboxes`.

The same dedicated-node warning applies. Startup `pkill` and stale-sandbox cleanup default off. The
exit trap stops Ray only after this launcher successfully starts its own head, and always attempts
run-owned sandbox cleanup. Preflight failure therefore does not stop a pre-existing Ray head, but
the launcher still requires port 8265 and the eight GPUs to be free. It can additionally delete
stale sandboxes recorded by its manifest when `CLEANUP_STALE_SANDBOXES=1`.

## Tests

These tests are offline and do not create sandboxes:

```bash
cd swe_harness
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/src:$PWD/external/azure-modal:$PWD/.." \
python -m pytest -q -p no:cacheprovider \
  tests/agents/test_testpatch_resolve.py \
  tests/scripts/repair/test_build_testpatch_testmap.py \
  tests/scripts/repair/test_selfrepair_gentest.py \
  tests/scripts/repair/test_selfrepair_f2p.py \
  tests/scripts/repair/test_oracle_f2p_full_repair_verify_loop.py
```

From the bundle root, validate the collected training path without GPUs or live sandboxes:

```bash
bash -n examples/mini_swe/scripts/train_swe_qwen3.5_35B_A3B_repair.sh
bash -n examples/mini_swe/scripts/train_resolve_scaleswe.sh

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/swe_harness/src:$PWD/swe_harness/external/azure-modal" \
python -m pytest -q -p no:cacheprovider \
  examples/mini_swe/test_f2p_repair_rollout.py \
  examples/mini_swe/test_repair_reward.py \
  examples/mini_swe/test_swe_reward.py \
  tests/test_gentest_native_rollout.py \
  tests/test_resolve_native_rollout.py
```

## Validation of this snapshot

On 2026-08-20, both the source checkout and a separately built staging bundle passed the offline
checks above:

- harness path: `102 passed`;
- training path: `157 passed, 2 skipped`;
- `bash -n` for both requested launchers;
- builder selection: 339 source files / 4,393,686 bytes;
- Python compilation, focused harness tests, and embedded-credential marker checks.

These checks did **not** launch GPUs, Ray, SGLang, a model server, W&B, or a live sandbox. No real
training step, behavior-contract generation, self-repair rollout, or official SWE-bench verification was run as
part of packaging this snapshot.

## Known boundaries

- The persistent generated-test verifier resets with `git reset --hard` and `git clean -fdq`;
  ignored caches/databases/build products can survive. Use fresh sandboxes or strengthen cleanup
  before claiming strict cross-candidate filesystem isolation.
- `batch_resolve.py` attempts git-history sanitization but does not validate or persist the returned
  status. The command guard is enabled by default, and blocked history-access attempts are visible
  only in trajectories. This snapshot alone cannot prove per-row sanitization success; a strict
  Gold-hidden claim needs an added checked/persisted status or an independent sandbox/trajectory
  audit. The map exporter does not enforce this boundary.
- The self-repair gate applies the behavior-contract patch and runs/parses its command through the repair path;
  it is not byte-for-byte the behavior-contract generation-time official-runner call path.
- `max_tests=1` matches a max-one map but truncates future multi-entry maps. Keep the map cap and
  repair cap aligned.
- The two training launchers are single-node, eight-GPU operational snapshots. A shell syntax pass,
  import pass, or CPU unit test does not prove that the model/checkpoint topology fits or that a
  rollout persisted successfully.
- Sandbox and W&B credentials supplied at runtime are forwarded through Ray job arguments/runtime
  environment and may be visible to Ray dashboard or log readers. Restrict dashboard/log access and
  retention accordingly.
- The source-repair launcher uses a Gold F2P signal. The resolve launcher defaults to V11 and uses
  reward-side candidate/Gold data. Keep both distinct from strict behavior-contract generation and from official
  SWE-bench resolution.
- The root slime code is Apache-2.0; `swe_harness/LICENSE.md` covers the MIT mini-swe-agent subtree.
  This snapshot also contains local research modifications. Review provenance, benchmark/data terms,
  third-party dependencies, ownership, and release approval before external redistribution.
- Importing the bundled mini-swe-agent can load user-level configuration from
  `~/.config/mini-swe-agent/.env`. Use an isolated home/config directory for release validation and
  inspect ambient environment variables before a run.
- The full slime runtime contains optional example and cleanup helpers beyond the two documented
  launchers; some are intentionally node-wide. Do not execute scripts by discovery. Inspect their
  cleanup scope, and use a dedicated node for either requested launcher.

## Refreshing the bundle from the source checkout

The source checkout keeps a manifest-driven builder at
`swe_harness/behavior_contract_selfrepair_share/build_bundle.py`. It copies current worktree bytes, not Git HEAD,
because several harness and training files are locally modified or untracked. It rejects selected
files containing common embedded private-key/W&B-key markers and skips result/run/trajectory/
checkpoint/cache directories. Output files are normalized to non-world-writable modes (`0755` for
shell/existing executables and `0644` otherwise).

```bash
python swe_harness/behavior_contract_selfrepair_share/build_bundle.py --check
python swe_harness/behavior_contract_selfrepair_share/build_bundle.py \
  --output dist/behavior_contract_selfrepair_training_YYYYMMDD
```

The builder refuses to overwrite an existing directory or archive.
