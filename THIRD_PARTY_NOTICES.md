# Third-party notices

This release contains a focused snapshot of code from multiple components. Preserve the notices and
license files when redistributing modified copies.

## slime

- Upstream: <https://github.com/THUDM/slime>
- License: Apache License 2.0
- License text: `LICENSE`

## mini-swe-agent subtree

- Upstream: <https://github.com/SWE-agent/mini-swe-agent>
- License: MIT
- License text and copyright notice: `swe_harness/LICENSE.md`
- This archive includes local research modifications.

## Azure-modal-compatible API

This release includes a focused compatibility client at
`swe_harness/external/azure-modal/client`. It is part of this source kit, is covered by the root
license, and implements the Sandbox Orchestrator 0.1.x HTTP contract used by the included code. It
is not a copy of the separately distributed `aks_modal` client and does not include the sandbox
server.

Megatron-LM, PyTorch, CUDA, Ray, SGLang, Transformer Engine, FlashInfer, FlashAttention, DeepEP,
checkpoints, and evaluation artifacts are external dependencies and are not included. The training
JSONL under `data/` derives from SWE-ReBench/SWE-rebench-V2, Scale-SWE, model-generated trajectories,
and many upstream repositories. The underlying benchmark, model-provider, and repository licenses
and terms continue to apply; see `DATA_CARD.md`.
