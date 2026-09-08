#!/usr/bin/env python3
"""
Delete leftover Azure CaaS sandboxes that were not cleaned up properly
(e.g. after Ctrl+C).

Reads sandbox marker files written by ``azure_modal_docker.py`` from the
manifest directory (default: ``examples/mini_swe/.azure_sandboxes/``),
deletes each sandbox via the CaaS API, and removes the marker file.

Usage:
    python scripts/cleanup_azure_sandboxes.py [--manifest-dir DIR]

Environment variables:
    AZURE_SANDBOX_MANIFEST_DIR  Override the default manifest directory.
    AZURE_CAAS_ENDPOINT         Fallback endpoint when a marker file has
                                no endpoint recorded.
    AZURE_MODAL_PATH            Path to the azure-modal client library.
"""

import argparse
import asyncio
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

_DEFAULT_MANIFEST_DIR = "/tmp/.azure_sandboxes"


def _read_records(manifest_dir: str) -> list[dict[str, str]]:
    """Read all sandbox records from the manifest directory."""
    manifest_dir = os.path.abspath(manifest_dir)
    if not os.path.isdir(manifest_dir):
        return []
    records = []
    for name in os.listdir(manifest_dir):
        marker = os.path.join(manifest_dir, name)
        try:
            with open(marker) as f:
                records.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            records.append({"sandbox_id": name, "endpoint": ""})
    return records


def _remove_marker(manifest_dir: str, sandbox_id: str) -> None:
    marker = os.path.join(manifest_dir, sandbox_id)
    try:
        os.remove(marker)
    except OSError:
        pass


async def cleanup(
    manifest_dir: str,
    fallback_endpoint: str | None = None,
    *,
    sandbox_prefix: str = "",
    request_timeout: int = 30,
) -> int:
    manifest_dir = os.path.abspath(manifest_dir)
    records = _read_records(manifest_dir)
    if sandbox_prefix:
        records = [record for record in records if record["sandbox_id"].startswith(sandbox_prefix)]
    if not records:
        print(f"No sandbox markers found in {manifest_dir}")
        return 0

    print(f"Found {len(records)} leftover sandbox(es) to clean up:")
    for r in records:
        print(f"  - {r['sandbox_id']}  endpoint={r.get('endpoint', 'N/A')}")

    # Lazy import
    azure_modal_path = os.environ.get("AZURE_MODAL_PATH")
    if azure_modal_path and azure_modal_path not in sys.path:
        sys.path.insert(0, azure_modal_path)
    from client import AsyncSandboxClient

    # Group by endpoint so we reuse one client per endpoint
    by_endpoint: dict[str, list[str]] = {}
    for r in records:
        ep = r.get("endpoint") or fallback_endpoint or ""
        if not ep:
            print(f"  WARNING: No endpoint for sandbox {r['sandbox_id']}, skipping. "
                  "Set AZURE_CAAS_ENDPOINT to provide a fallback.")
            continue
        by_endpoint.setdefault(ep, []).append(r["sandbox_id"])

    failures = 0
    for endpoint, sandbox_ids in by_endpoint.items():
        client = AsyncSandboxClient(endpoint, timeout=request_timeout, auto_cleanup=False)
        try:
            async def delete_one(sid: str) -> bool:
                try:
                    await client.delete_sandbox(sid)
                    print(f"  Deleted {sid}")
                    _remove_marker(manifest_dir, sid)
                    return True
                except Exception as exc:
                    print(f"  Failed to delete {sid}: {exc}")
                    return False

            results = await asyncio.gather(*(delete_one(sid) for sid in sandbox_ids))
            failures += sum(not result for result in results)
        finally:
            try:
                if hasattr(client, "close"):
                    await client.close(cleanup=False)
                elif hasattr(client, "_session") and client._session and not client._session.closed:
                    await client._session.close()
            except Exception:
                pass

    skipped = sum(1 for record in records if not (record.get("endpoint") or fallback_endpoint))
    failures += skipped
    print(f"Cleanup complete: deleted={len(records) - failures}, failed_or_skipped={failures}")
    return failures


def main():
    parser = argparse.ArgumentParser(
        description="Delete leftover Azure CaaS sandboxes from a previous run."
    )
    parser.add_argument(
        "--manifest-dir",
        default=os.environ.get("AZURE_SANDBOX_MANIFEST_DIR", _DEFAULT_MANIFEST_DIR),
        help="Directory containing sandbox marker files "
             "(default: /tmp/.azure_sandboxes/ or AZURE_SANDBOX_MANIFEST_DIR)",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Only delete sandbox IDs beginning with this prefix.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=30,
        help="Per-request cleanup timeout in seconds (default: 30).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    fallback_endpoint = os.environ.get("AZURE_CAAS_ENDPOINT")
    failures = asyncio.run(
        cleanup(
            args.manifest_dir,
            fallback_endpoint,
            sandbox_prefix=args.prefix,
            request_timeout=args.request_timeout,
        )
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
