"""Phase 2 (docs/DECISIONS.md ADR-010): runs scripts/audit_multilingual_memory.py as an isolated
subprocess per size (matching R-032's methodology -- each measurement gets a fresh interpreter, no
cross-run contamination) and aggregates the results into one file.

Usage: python scripts/run_multilingual_memory_audit.py --size 100k --size 150k --size 200k
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", action="append", required=True, choices=["100k", "150k", "200k"])
    args = parser.parse_args()

    results = {}
    for size_name in args.size:
        index_dir = REPO_ROOT / "data" / "index" / f"multilingual_{size_name}"
        print(f"Auditing {size_name} in an isolated subprocess...")
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "audit_multilingual_memory.py"),
             "--index-dir", str(index_dir)],
            capture_output=True, text=True, check=True, cwd=str(REPO_ROOT),
        )
        result = json.loads(proc.stdout.strip().splitlines()[-1])
        results[size_name] = result
        print(json.dumps(result, indent=2, ensure_ascii=False))

    out_path = REPO_ROOT / "eval" / "multilingual_memory_audit.json"
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    existing.update(results)
    out_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote memory audit -> {out_path}")
