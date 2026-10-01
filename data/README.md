# ExecCritic training data

This directory contains the released Test-agent SFT data and the two role-specific RL prompt
datasets. Files are ordered, line-aligned JSONL shards kept below GitHub's per-file size limit.

| Dataset | Rows | Parts | Concatenated SHA-256 |
|---|---:|---:|---|
| `sft/test_agent_deepseek_v4_flash_0731_cot` | 6,257 | 7 | `69032413746d98d1948d79ef6dfeb8fb09bff68db20ece38d22ec6abf786a5e8` |
| `rl/test_agent` | 1,822 | 2 | `43e4339045be5b3cdfad2e8311d4b94e470b0c3dbe718b3889db02419c7ab0a7` |
| `rl/repair_agent` | 1,435 | 1 | `4230c642ffa7e1b2917bcd6cef8832cb2c8d2e9dde8b88fab937ce80ef6edf16` |

Validate every shard, schema, row count, hash, and the release safety scan:

```bash
python data/validate_data.py
```

Reassemble a dataset when a trainer requires one path:

```bash
cat data/sft/test_agent_deepseek_v4_flash_0731_cot/train-*-of-*.jsonl > /tmp/test_agent_sft.jsonl
cat data/rl/test_agent/train-*-of-*.jsonl > /tmp/test_agent_rl.jsonl
cat data/rl/repair_agent/train-*-of-*.jsonl > /tmp/repair_agent_rl.jsonl
```

Use the files in lexical order. See [`manifest.json`](manifest.json) for byte-exact provenance and
[`../DATA_CARD.md`](../DATA_CARD.md) for schemas, composition, transformations, and limitations.
