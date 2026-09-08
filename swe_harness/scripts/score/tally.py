#!/usr/bin/env python3
"""Per-run SWE-V tally — folded into the end of every verify launcher.
Usage: tally.py <out_dir> [denom]
  globs <out_dir>/**/verify_results*.json, counts resolved, prints pass@1.
  denom: pass@1 denominator (default = full N if known, else #found).
         SWE-V convention = missing+unscoreable count as unresolved over the
         intended N (e.g. 500 / 300), so pass an explicit denom for headline.
Per-category breakdown printed if the difficulty CSV exists."""
import json, glob, sys, os, csv
from collections import Counter
from pathlib import Path

if len(sys.argv) < 2:
    print("usage: tally.py <out_dir> [denom]"); sys.exit(2)
out_dir = sys.argv[1]
denom = int(sys.argv[2]) if len(sys.argv) > 2 else None

res = {}
pats = [f"{out_dir}/**/verify_results*.json", f"{out_dir}/*/verify_results*.json", f"{out_dir}/verify_results*.json"]
files = []
for p in pats:
    files += glob.glob(p, recursive=True)
for vf in sorted(set(files)):
    try:
        d = json.load(open(vf))
    except Exception:
        continue
    for iid, v in d.items():
        if isinstance(v, dict):
            res[iid] = v.get("resolved") is True
n = len(res); solved = sum(res.values())
D = denom or n or 1
print(f"[tally] dir={out_dir}")
print(f"[tally] verified={n} resolved={solved} pass@1={100*solved/D:.2f}% (denom={D})")

# optional per-category
csvp = os.environ.get(
    "DIFFICULTY_CSV",
    str(Path(__file__).resolve().parents[2] / "eval_logs" / "sft_difficulty_categories.csv"),
)
if os.path.exists(csvp):
    cat = {r["instance_id"]: r["category"] for r in csv.DictReader(open(csvp))}
    tot, hit = Counter(), Counter()
    for iid, r in res.items():
        c = cat.get(iid, "?"); tot[c] += 1; hit[c] += r
    for c in ["hard", "medium", "easy", "resolved"]:
        if tot[c]:
            print(f"[tally]   {c:>9}: {hit[c]}/{tot[c]} = {100*hit[c]/tot[c]:.1f}%")
