#!/usr/bin/env python3
"""Validate the released ExecCritic JSONL shards against manifest.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SENSITIVE_PATTERNS = {
    "private key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    "AWS access key": re.compile(
        rb"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"
    ),
    "Azure SAS signature": re.compile(rb"[?&]sig=[A-Za-z0-9%/+_-]{20,}", re.I),
    "GitHub token": re.compile(
        rb"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{40,})"
    ),
    "bearer JWT": re.compile(
        rb"Bearer\s+eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
        re.I,
    ),
    "release-host path": re.compile(rb"/data/users/(?:leitian|wenlinyao)/"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"validation failed: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("manifest.json"),
    )
    parser.add_argument(
        "--skip-sensitive-scan",
        action="store_true",
        help="skip the release-host path and high-confidence credential scan",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    repo_root = manifest_path.parent.parent
    manifest = json.loads(manifest_path.read_text())

    for dataset_name, dataset in manifest["datasets"].items():
        combined = hashlib.sha256()
        total_bytes = 0
        total_rows = 0
        required = set(dataset["required_top_level_keys"])

        for part in dataset["parts"]:
            path = repo_root / part["path"]
            if not path.is_file():
                fail(f"{dataset_name}: missing {part['path']}")
            if path.stat().st_size != part["bytes"]:
                fail(f"{dataset_name}: byte count differs for {part['path']}")
            if sha256_file(path) != part["sha256"]:
                fail(f"{dataset_name}: SHA-256 differs for {part['path']}")

            part_rows = 0
            with path.open("rb") as handle:
                for part_rows, line in enumerate(handle, 1):
                    combined.update(line)
                    total_bytes += len(line)
                    if not args.skip_sensitive_scan:
                        for label, pattern in SENSITIVE_PATTERNS.items():
                            if pattern.search(line):
                                fail(
                                    f"{dataset_name}: {label} match in "
                                    f"{part['path']}:{part_rows}"
                                )
                    try:
                        row = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        fail(f"{dataset_name}: invalid JSON in {part['path']}:{part_rows}: {exc}")
                    missing = required.difference(row)
                    if missing:
                        fail(
                            f"{dataset_name}: {part['path']}:{part_rows} missing "
                            f"{sorted(missing)}"
                        )

            if part_rows != part["rows"]:
                fail(f"{dataset_name}: row count differs for {part['path']}")
            total_rows += part_rows

        if total_bytes != dataset["bytes"]:
            fail(f"{dataset_name}: combined byte count differs")
        if total_rows != dataset["rows"]:
            fail(f"{dataset_name}: combined row count differs")
        if combined.hexdigest() != dataset["sha256"]:
            fail(f"{dataset_name}: combined SHA-256 differs")
        print(
            f"{dataset_name}: ok; rows={total_rows}; bytes={total_bytes}; "
            f"sha256={combined.hexdigest()}"
        )


if __name__ == "__main__":
    main()
