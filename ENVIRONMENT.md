# Runtime environment

The archive is a source release, not a hermetic GPU environment. `requirements-test.txt` adds the
harness and focused test dependencies; it does not install the slime runtime or pin PyTorch,
Transformers, Pillow, SGLang, or the CUDA stack.

GPU training additionally requires a mutually compatible set of:

- eight visible CUDA GPUs for the supplied single-node launchers;
- PyTorch and CUDA libraries;
- Megatron-LM, supplied through `MEGATRON_ROOT`;
- Ray;
- SGLang and its matching kernel/FlashInfer stack;
- the bundled Azure-modal-compatible client (override it through `AZURE_MODAL_DIR` for gentest or
  `AZURE_MODAL_PATH` for repair when an authorized external client is required);
- Transformer Engine, DeepEP, FlashAttention or flash-linear-attention as required by the model;
- the exact Hugging Face and torch-distributed checkpoints selected by the launcher.

Versions of those native components are intentionally not inferred or installed by this archive.
For a reproducible run, record the container image by immutable digest, driver/CUDA versions,
Megatron-LM commit, SGLang/kernel/FlashInfer versions, checkpoint hashes, prompt-data hash and row
count, effective environment, and rendered launcher command.

For offline validation, start inside a compatible slime runtime/image. If its Python packages are
available system-wide, an optional isolated environment can inherit them:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-test.txt
```

A fresh CPU-only environment created from this requirements file alone is not sufficient for the
bundled imports. Record the base image and already-installed runtime versions used for validation.

The code was designed for remote Azure-modal-compatible sandboxes. A focused compatibility client
for the Sandbox Orchestrator 0.1.x API is bundled; an authorized external client can be selected
through the launcher path overrides. Supply endpoint and credentials only at runtime. Do not store
them in `ENV.example`, shell scripts, Git history, Ray runtime metadata retained beyond the run, or
release artifacts.

Before scaling out, validate `/v1/models`, one real model completion, one sandbox create/execute/
delete lifecycle, one persisted rollout, reward output, and cleanup. A process, GPU allocation, or
shell syntax pass alone is not a successful training smoke.
