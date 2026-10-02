"""Run the benchmark twice in independent processes and compare content hashes.

Gate G5 requires that two runs of the same configuration agree bit for bit.
Running them in the *same* process would prove nothing -- module-level state,
warm caches and RNG carry-over could mask a real difference -- so this script
deliberately spawns two fresh interpreters.

On mismatch it reports the *path* of every differing field rather than only
that the digests differ; a bare "hashes differ" gives a reader nothing to act
on.

Usage::

    python scripts/check_determinism.py --profile smoke --datasets D1_clean
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run_once(out_path: str, profile: str, datasets: str, seeds: str, systems: str) -> dict:
    """Execute one benchmark run in a fresh interpreter and return the document."""
    cmd = [sys.executable, "-m", "voiceforge.cli", "--profile", profile, "run",
           "--out", out_path, "--no-fail-gates"]
    if datasets:
        cmd += ["--datasets", datasets]
    if seeds:
        cmd += ["--seeds", seeds]
    if systems:
        cmd += ["--systems", systems]
    env = {**os.environ, "PYTHONPATH": ROOT, "PYTHONHASHSEED": "0"}
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env, check=False)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"benchmark run failed with exit code {proc.returncode}")
    with open(out_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def report_differences(a: dict, b: dict) -> list[str]:
    """Return a list of ``path: x vs y`` strings for every differing field."""
    from voiceforge.eval.benchmark import _hashable

    left, right = _hashable(a), _hashable(b)
    out: list[str] = []

    def walk(x, y, path: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y)):
                walk(x.get(key), y.get(key), f"{path}.{key}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                out.append(f"{path}: length {len(x)} vs {len(y)}")
                return
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
        elif x != y:
            out.append(f"{path}: {x!r:.80} vs {y!r:.80}")

    walk(left, right, "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify gate G5 (bit-exact determinism).")
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--datasets", default="D1_clean")
    ap.add_argument("--seeds", default="17")
    ap.add_argument("--systems", default="")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        a = run_once(os.path.join(tmp, "a.json"), args.profile, args.datasets, args.seeds, args.systems)
        b = run_once(os.path.join(tmp, "b.json"), args.profile, args.datasets, args.seeds, args.systems)

    ha, hb = a.get("content_hash"), b.get("content_hash")
    print(f"run A content_hash: {ha}")
    print(f"run B content_hash: {hb}")

    if ha == hb:
        n_cells = len(a.get("results", []))
        print(f"\n[OK] G5 determinism: two independent processes agree ({n_cells} result cells)")
        return 0

    print("\n[FAIL] G5 determinism: content hashes differ")
    diffs = report_differences(a, b)
    if not diffs:
        print("  (no differing fields found -- the hash itself is unstable)")
    for line in diffs[:40]:
        print(f"  {line}")
    if len(diffs) > 40:
        print(f"  ... and {len(diffs) - 40} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
