#!/usr/bin/env python3
"""Compare pytest-benchmark JSON results against stored ratios and a previous run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_means(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    means: dict[str, float] = {}
    for benchmark in payload.get("benchmarks", []):
        name = benchmark.get("name")
        stats = benchmark.get("stats") or {}
        mean = stats.get("mean")
        if name is None or mean is None:
            continue
        means[name] = float(mean)
    return means


def check_ratios(means: dict[str, float], ratios: list[dict]) -> list[str]:
    errors: list[str] = []
    for spec in ratios:
        slower_name = spec["slower"]
        faster_name = spec["faster"]
        min_ratio = float(spec["min_ratio"])
        if slower_name not in means:
            errors.append(f"missing slower benchmark {slower_name!r}")
            continue
        if faster_name not in means:
            errors.append(f"missing faster benchmark {faster_name!r}")
            continue
        faster = means[faster_name]
        if faster <= 0:
            errors.append(f"{faster_name} has non-positive mean {faster}")
            continue
        ratio = means[slower_name] / faster
        if ratio < min_ratio:
            errors.append(
                f"{spec.get('description', slower_name)}: "
                f"{slower_name}/{faster_name} = {ratio:.2f}x "
                f"(required >= {min_ratio:.2f}x)"
            )
    return errors


def check_previous(
    current: dict[str, float], previous: dict[str, float], max_slowdown: float
) -> list[str]:
    errors: list[str] = []
    for name, previous_mean in previous.items():
        if name not in current or previous_mean <= 0:
            continue
        slowdown = current[name] / previous_mean
        if slowdown > max_slowdown:
            errors.append(
                f"{name}: {slowdown:.2f}x slower than previous mean "
                f"({current[name]:.4f}s vs {previous_mean:.4f}s, max {max_slowdown:.2f}x)"
            )
    return errors


def format_ratio_table(means: dict[str, float], ratios: list[dict]) -> str:
    lines = [
        "| Slower | Faster | Ratio | Required |",
        "|---|---|---:|---:|",
    ]
    for spec in ratios:
        slower_name = spec["slower"]
        faster_name = spec["faster"]
        if slower_name not in means or faster_name not in means:
            ratio_text = "n/a"
        elif means[faster_name] <= 0:
            ratio_text = "n/a"
        else:
            ratio_text = f"{means[slower_name] / means[faster_name]:.1f}x"
        lines.append(
            f"| `{slower_name}` | `{faster_name}` | {ratio_text} | {spec['min_ratio']:.1f}x |"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--ratios", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=None)
    parser.add_argument("--max-slowdown", type=float, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    baseline = json.loads(args.ratios.read_text(encoding="utf-8"))
    means = load_means(args.current)
    ratios = baseline.get("same_run_ratios", [])
    errors = check_ratios(means, ratios)

    previous_path = args.previous
    max_slowdown = args.max_slowdown
    if max_slowdown is None:
        max_slowdown = float(baseline.get("previous_run", {}).get("max_slowdown", 2.0))
    if previous_path is not None and previous_path.is_file():
        errors.extend(check_previous(means, load_means(previous_path), max_slowdown))
    elif previous_path is not None:
        print(
            f"No previous benchmark file at {previous_path}; skipping regression compare"
        )

    report = ["## Benchmark comparison", "", format_ratio_table(means, ratios), ""]
    if errors:
        report.append("### Failures")
        report.extend(f"- {error}" for error in errors)
    else:
        report.append("All stored ratio and regression checks passed.")
    text = "\n".join(report) + "\n"
    print(text)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(text)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
