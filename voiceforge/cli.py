"""Command line interface.

Subcommands
-----------
``config``    print the resolved configuration (honours ``ENV_VOICEFORGE_*``)
``schema``    print the full declared schema
``run``       execute the benchmark grid and write ``benchmark.json``
``report``    render a report from an existing ``benchmark.json``
``ablate``    run the ablation study
``selftest``  run the invariant checks that do not need the full grid

Exit codes: ``0`` success, ``1`` a gate or runtime failure, ``2`` a usage or
configuration error.  The distinction matters in CI: a smoke run that *fails a
gate* must not look like a run that *crashed*.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

from .core.config import DEFAULT_SCHEMA, SCHEMA_GROUPS, describe_schema, load_config
from .core.errors import VoiceForgeError
from .core.logging import configure_logging, get_logger, setup_stdout
from .core.memory import current_rss_mb, format_mb, peak_rss_mb
from .core.seed import preset_threads

__all__ = ["main", "build_parser"]

log = get_logger("cli")

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (kept separate so tests can introspect it)."""
    p = argparse.ArgumentParser(
        prog="voiceforge",
        description="VoiceForge -- a deterministic speaker verification benchmark (author: 晨星).",
    )
    p.add_argument("--profile", choices=("smoke", "demo", "bench"), default=None, help="run profile")
    p.add_argument("--seed", type=int, default=None, help="override the master seed")
    p.add_argument("--out-dir", default=None, help="override the artifact directory")
    p.add_argument("--verbose", action="store_true", help="verbose stage logging")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config key, e.g. --set gmm.n_gauss=64")

    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="print the resolved configuration")
    sub.add_parser("schema", help="print the declared schema")

    run = sub.add_parser("run", help="run the benchmark grid")
    run.add_argument("--datasets", default=None, help="comma separated dataset ids")
    run.add_argument("--systems", default=None, help="comma separated system ids")
    run.add_argument("--seeds", default=None, help="comma separated seeds")
    run.add_argument("--out", default=None, help="output path for benchmark.json")
    run.add_argument("--no-fail-gates", action="store_true",
                     help="always exit 0, even when a gate fails (for exploration)")

    rep = sub.add_parser("report", help="render a report from benchmark.json")
    rep.add_argument("path", help="path to benchmark.json")

    abl = sub.add_parser("ablate", help="run the ablation study")
    abl.add_argument("--datasets", default=None, help="comma separated dataset ids")
    abl.add_argument("--switches", default=None, help="comma separated ablation switches")
    abl.add_argument("--out", default=None, help="output path for the ablation JSON")

    sub.add_parser("selftest", help="run fast invariant checks")
    sub.add_parser("env", help="print the resolved backend and environment")
    return p


def _parse_overrides(items: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Turn ``--set gmm.n_gauss=64`` into ``{"gmm": {"n_gauss": "64"}}``."""
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        if "." not in key:
            raise ValueError(f"--set expects a dotted key like gmm.n_gauss, got {key!r}")
        group, _, name = key.partition(".")
        out.setdefault(group, {})[name] = value
    return out


def _split(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _seeds(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _build_config(args: argparse.Namespace):
    """Resolve the configuration from profile + --set + --seed + --out-dir."""
    overrides = _parse_overrides(args.set)
    if args.seed is not None:
        overrides.setdefault("app", {})["seed"] = args.seed
    if args.out_dir is not None:
        overrides.setdefault("app", {})["out_dir"] = args.out_dir
    return load_config(args.profile, overrides)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    # Threads must be pinned before numpy is imported anywhere downstream.
    preset_threads(1)
    setup_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        config = _build_config(args)
    except (VoiceForgeError, ValueError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        return _dispatch(args, config)
    except VoiceForgeError as exc:
        print(f"[{exc.code}] {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return EXIT_USAGE


def _dispatch(args: argparse.Namespace, config) -> int:
    from .core.config import config_to_dict

    if args.command == "config":
        print(json.dumps(config_to_dict(config), ensure_ascii=False, indent=2, sort_keys=True))
        return EXIT_OK

    if args.command == "schema":
        rows = describe_schema()
        print(f"{'group':<10} {'key':<22} {'type':<7} {'default':<14} doc")
        print("-" * 100)
        for row in rows:
            default = row["default"]
            if isinstance(default, (list, tuple)):
                default = ",".join(str(x) for x in default)
            print(f"{row['group']:<10} {row['key']:<22} {row['type']:<7} {str(default)[:14]:<14} {row['doc']}")
        return EXIT_OK

    if args.command == "env":
        from .core.backend import describe_backend

        print(json.dumps(describe_backend(), ensure_ascii=False, indent=2, sort_keys=True))
        return EXIT_OK

    if args.command == "selftest":
        return _selftest(config)

    if args.command == "run":
        return _run(args, config)

    if args.command == "report":
        return _report(args.path, config)

    if args.command == "ablate":
        return _ablate(args, config)

    parser = build_parser()
    parser.error(f"unknown command {args.command!r}")
    return EXIT_USAGE


def _selftest(config) -> int:
    """Fast invariant checks that do not need the full grid."""
    from .core.backend import available_librosa, resolve_dsp
    from .core.seed import content_hash
    from .sv.frontend import dct_matrix, mel_filterbank

    checks: list[tuple[str, bool, str]] = []

    ok, detail = available_librosa()
    info = resolve_dsp("auto")
    checks.append(("librosa probe", True, f"available={ok} detail={detail} tier={info.resolved}"))
    checks.append(("threads pinned", os.environ.get("OMP_NUM_THREADS") == "1",
                   f"OMP={os.environ.get('OMP_NUM_THREADS')} MKL_CBWR={os.environ.get('MKL_CBWR')}"))

    t0 = mel_filterbank(16000, 512, 40, 20.0, 7600.0, backend="tier0")
    t1 = mel_filterbank(16000, 512, 40, 20.0, 7600.0, backend="tier1")
    delta = float(abs(t0.matrix - t1.matrix).max()) if ok else float("nan")
    checks.append(("tier0/tier1 mel parity", (not ok) or delta < 1e-7, f"max|diff|={delta:.3e}"))

    d = dct_matrix(20, 40)
    orth = float(abs(d @ d.T - __import__("numpy").eye(20)).max())
    checks.append(("DCT orthonormality", orth < 1e-12, f"max|err|={orth:.3e}"))

    h1 = content_hash({"a": 1, "b": [2, 3]})
    h2 = content_hash({"b": [2, 3], "a": 1})
    checks.append(("content hash stable", h1 == h2, f"{h1} == {h2}"))

    rss = peak_rss_mb()
    checks.append(("rss probe", rss > 0.0, f"peak={format_mb(rss)} current={format_mb(current_rss_mb())}"))

    width = max(len(name) for name, _, _ in checks) + 2
    print(f"{'check':<{width}} {'status':<8} detail")
    print("-" * 88)
    failed = 0
    for name, passed, detail in checks:
        print(f"{name:<{width}} {'[OK]' if passed else '[FAIL]':<8} {detail}")
        failed += 0 if passed else 1
    print("-" * 88)
    print(f"{len(checks) - failed}/{len(checks)} checks passed")
    return EXIT_OK if failed == 0 else EXIT_GATE_FAILED


def _run(args: argparse.Namespace, config) -> int:
    from .eval.benchmark import run_benchmark
    from .eval.report import evaluate_gates
    from .eval.report import print_report

    out_path = args.out or os.path.join(config.app.out_dir, "benchmark.json")
    doc = run_benchmark(
        config,
        systems=_split(args.systems),
        datasets=_split(args.datasets),
        seeds=_seeds(args.seeds),
        out_path=out_path,
    )
    print_report(doc, config)

    gates = evaluate_gates(doc, config)["gates"]
    failed = [g for g, e in gates.items() if e["status"] == "fail"]
    unevaluated = [g for g, e in gates.items() if e["status"] == "not_evaluated"]
    print(f"\nbenchmark written to {out_path}")
    print(f"gates: {len(gates) - len(failed) - len(unevaluated)} pass, {len(failed)} fail, "
          f"{len(unevaluated)} not evaluated")
    if unevaluated:
        print(f"  not evaluated: {', '.join(sorted(unevaluated))}")
    if failed and not args.no_fail_gates:
        return EXIT_GATE_FAILED
    return EXIT_OK


def _report(path: str, config) -> int:
    from .eval.benchmark import read_benchmark
    from .eval.report import print_report

    if not os.path.isfile(path):
        print(f"no such benchmark file: {path}", file=sys.stderr)
        return EXIT_USAGE
    doc = read_benchmark(path)
    print_report(doc, config)
    return EXIT_OK


def _ablate(args: argparse.Namespace, config) -> int:
    from .eval.ablation import run_ablations

    result = run_ablations(
        config,
        datasets=_split(args.datasets),
        switches=_split(args.switches),
    )
    out = args.out or os.path.join(config.app.out_dir, "ablations.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
    print(f"ablation results written to {out}")
    base = result.get("baseline", {}).get("mean")
    print(f"{'switch':<18} {'baseline':>10} {'ablated':>10} {'delta':>10} {'expected':>10} {'ok':>5}")
    print("-" * 70)
    for row in result.get("rows", []):
        print(
            f"{row['switch']:<18} {row['baseline_mean_eer']:>10.4f} {row['ablated_mean_eer']:>10.4f} "
            f"{row['observed_delta_eer']:>+10.4f} {row['expected_direction']:>10} "
            f"{'yes' if row['direction_as_expected'] else 'NO':>5}"
        )
    for w in result.get("warnings", []):
        print(f"  warning: {w.get('kind')} {w.get('switch', '')}")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
