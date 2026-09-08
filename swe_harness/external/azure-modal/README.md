# Bundled Sandbox Orchestrator compatibility client

This directory contains the small client used by the release's Azure-compatible
sandbox environments. It targets the `Sandbox Orchestrator` 0.1.x HTTP API and
supports both synchronous and asynchronous callers.

The directory itself is added to `PYTHONPATH`; callers import `client` or
`client.sandbox_client`. Runtime configuration is provided through:

- `SANDBOX_BASE_URL`
- `SANDBOX_API_KEY` (`X-API-Key` on authenticated requests)
- `SANDBOX_PREFIX` (optional sandbox-ID prefix)
- `AZURE_SANDBOX_MANIFEST_DIR` (optional durable cleanup markers)

The implementation is part of this source kit and covered by the root project
license. It is not the separately distributed `aks_modal` package and does not
contain the server implementation.
