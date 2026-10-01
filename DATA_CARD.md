# ExecCritic training data card

## Summary

This release provides three JSONL datasets used by the separate ExecCritic Test and Repair training
paths. They are training inputs, not evaluation results, model checkpoints, or evidence that a
particular checkpoint completed training.

| Dataset | Purpose | Rows | Release bytes |
|---|---|---:|---:|
| Test-agent SFT | Off-policy test-construction trajectories | 6,257 | 502,294,450 |
| Test-agent RL | Candidate-patch-classification GRPO prompts | 1,822 | 106,862,417 |
| Repair-agent RL | Single-F2P feedback-conditioned repair prompts | 1,435 | 19,040,191 |

The exact ordered shards, original-artifact hashes, release hashes, row counts, and byte counts are
recorded in [`data/manifest.json`](data/manifest.json). Run `python data/validate_data.py` before
training.

## Dataset details

### Test-agent SFT

Every row contains `messages`, `metadata`, and a Bash-only `tools` schema. All 6,257 trajectories
record `openai/DeepSeek-V4-Flash-0731` as the source model, native visible reasoning, and the V11
resolve-style test-patch protocol. Admission required strict Base-fail/Gold-pass validation in the
retained source artifact. Multiple trajectories may share an instance ID; this release preserves
all source rows and does not claim instance-level deduplication.

### Test-agent RL

Every row contains `id`, `text`, `problem_statement`, `patch`, and `metadata`. The 1,822 training
prompts comprise 307 Scale-SWE rows and 1,515 SWE-rebench-V2 rows. Each retains four positive and
four negative candidate patches with unique patch hashes. Gold and candidate labels are reward-side
metadata and must remain outside the model-visible generation context.

### Repair-agent RL

Every row contains `problem_statement`, the Round-0 source `patch`, and `metadata`. The 1,435 rows
cover 711 repositories from the SWE-ReBench low-pass-rate pool. Each row contains exactly one
selected fail-to-pass test for feedback-conditioned Repair-agent training. These selected training
tests are privileged reward/feedback inputs; they are not deployment-time generated tests.

## Release transformations

The original row order and row counts are preserved. The public artifacts apply only these safety
transformations:

- remove the SFT build-host `metadata.source_path` field;
- remove Test-agent RL `source_file`, `reward_file`, and local subset-source provenance paths;
- replace high-confidence credential-shaped strings found in public repository trajectory content
  with explicit `<REDACTED_...>` markers.

The redactions comprise three AWS access-key-shaped strings, two AWS secret-key-shaped strings,
seven complete private-key blocks, six unmatched private-key markers, and four SAS-signature-shaped
strings. One AWS secret-key-shaped string occurs in the Repair-agent RL data; the remaining
credential-shaped strings occur in the SFT trajectories. The validator fails closed if these
credential forms or release-host paths reappear.

## Limitations and intended use

- The data is intended for research on repository-level test generation and software repair.
- Validation labels depend on the benchmark snapshots, containers, tests, and harness behavior used
  to construct the rows. They should not be treated as timeless ground truth.
- Training data can contain imperfect model reasoning, noisy tool output, long repository excerpts,
  and duplicated instances.
- Test-agent reward metadata contains information that must not be exposed to the generation policy.
- Passing a generated local test is not equivalent to passing the benchmark's full official evaluator.

## Licensing and attribution

The repository's Apache-2.0 license covers the release-specific code, packaging, and documentation.
Rows derive from SWE-ReBench/SWE-rebench-V2 and Scale-SWE tasks, model-generated trajectories, and
excerpts or patches from many upstream open-source repositories. Those underlying materials retain
their original licenses and terms. Users are responsible for reviewing the applicable benchmark,
model-provider, and upstream repository terms and preserving required attribution.
