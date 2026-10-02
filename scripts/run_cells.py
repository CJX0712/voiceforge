"""Run the 4-cell pipeline ablation ({LDA} x {centring}) and print the EER table.

Cells::

    A  no LDA, no centring   (the pre-change state)
    B  no LDA, centring
    C  LDA,    no centring
    D  LDA,    centring       (the full Dehak pipeline)

Usage::

    python scripts/run_cells.py --profile demo --out artifacts/cells.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="demo")
    ap.add_argument("--datasets", default="D1_clean,D2_white5,D3_tel8,D4_rev0,D5_short1,D6_mixneg")
    ap.add_argument("--seeds", default="17,29,41")
    ap.add_argument("--speakers", type=int, default=24)
    ap.add_argument("--utts", type=int, default=4)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--out", default="artifacts/cells.json")
    args = ap.parse_args()

    from voiceforge.core.config import load_config
    from voiceforge.core.logging import setup_stdout
    from voiceforge.eval.cells import run_cell_ablation

    setup_stdout()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg = load_config(args.profile, use_env=False).evolve(
        data={"n_speakers": args.speakers, "n_utts_per_speaker": args.utts, "use_cache": False},
        bench={"n_trials": args.trials},
    )
    result = run_cell_ablation(cfg, datasets=datasets, seeds=seeds)
    result["config"] = {
        "profile": args.profile,
        "n_speakers": args.speakers,
        "n_utts": args.utts,
        "n_trials": args.trials,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=str)

    print("\n=== 4-cell EER (mean +/- std over seeds) ===")
    systems = ["mfcc_cos_nbc", "gmmubm_map_cos", "ivector_plda_full"]
    header = f"{'cell':<20}" + "".join(f"{s:>26}" for s in systems)
    print(header)
    print("-" * len(header))
    for cell_id, cell in result["cells"].items():
        line = f"{cell_id:<20}"
        for s in systems:
            stat = cell["aggregate"].get(s)
            line += f"{stat['mean']:>18.4f}+-{stat['std']:<6.4f}" if stat else f"{'--':>26}"
        print(line)

    print("\n=== cosine diagnostics ===")
    print(f"{'cell':<20}{'raw_within':>12}{'raw_between':>13}{'cen_within':>12}{'cen_between':>13}{'raw_gap':>10}{'cen_gap':>10}")
    print("-" * 90)
    for cell_id, cell in result["cells"].items():
        c = cell.get("cosines_mean", {})
        print(
            f"{cell_id:<20}{c.get('raw_within_cosine', float('nan')):>12.4f}"
            f"{c.get('raw_between_cosine', float('nan')):>13.4f}"
            f"{c.get('centered_within_cosine', float('nan')):>12.4f}"
            f"{c.get('centered_between_cosine', float('nan')):>13.4f}"
            f"{c.get('raw_gap', float('nan')):>10.4f}{c.get('centered_gap', float('nan')):>10.4f}"
        )

    print("\n=== model geometry (first run of each cell) ===")
    print(f"{'cell':<20}{'lda_dim':>9}{'tv_dim':>8}{'kept':>6}{'plda_dim':>10}{'tr_sw':>10}{'tr_sn':>10}{'ubm':>6}")
    print("-" * 80)
    for cell_id, cell in result["cells"].items():
        g = (cell.get("geometry") or [{}])[0]
        def fmt(v, spec="%.2f"):
            return spec % v if isinstance(v, (int, float)) else "n/a"
        print(
            f"{cell_id:<20}{fmt(g.get('lda_dim'), '%d'):>9}{fmt(g.get('tv_dim'), '%d'):>8}"
            f"{fmt(g.get('n_kept_components'), '%d'):>6}{fmt(g.get('plda_dim'), '%d'):>10}"
            f"{fmt(g.get('trace_sigma_w')):>10}{fmt(g.get('trace_sigma_n')):>10}"
            f"{fmt(g.get('ubm_gauss'), '%d'):>6}"
        )

    inv = result["invariants"]
    print("\n=== INV-1..4 ===")
    for key in ("INV-1", "INV-2", "INV-3", "INV-4"):
        e = inv[key]
        print(f"  {key}  {e['status'].upper():<5} {e['hits']}/{e['total']}   {e['statement']}")
    print(f"\noverall: {inv['overall'].upper()}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
