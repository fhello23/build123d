"""Tests for the benchmark comparison helper."""

from __future__ import annotations

import json
from pathlib import Path

from tools.compare_benchmarks import check_previous, check_ratios, load_means, main


def _write_benchmarks(path: Path, means: dict[str, float]) -> None:
    payload = {
        "benchmarks": [
            {"name": name, "stats": {"mean": mean}} for name, mean in means.items()
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_means(tmp_path: Path):
    path = tmp_path / "current.json"
    _write_benchmarks(path, {"test_a": 2.0, "test_b": 0.1})
    assert load_means(path) == {"test_a": 2.0, "test_b": 0.1}


def test_check_ratios_passes_and_fails():
    means = {
        "test_sequential_fuse_284": 1.8,
        "test_batched_fuse_284": 0.02,
    }
    spec = [
        {
            "slower": "test_sequential_fuse_284",
            "faster": "test_batched_fuse_284",
            "min_ratio": 10.0,
        }
    ]
    assert check_ratios(means, spec) == []
    spec[0]["min_ratio"] = 200.0
    errors = check_ratios(means, spec)
    assert len(errors) == 1
    assert "required >= 200.00x" in errors[0]


def test_check_previous_slowdown():
    current = {"test_a": 2.1}
    previous = {"test_a": 1.0}
    assert check_previous(current, previous, max_slowdown=2.0) != []
    assert check_previous(current, previous, max_slowdown=3.0) == []


def test_main_writes_summary(tmp_path: Path, monkeypatch):
    current = tmp_path / "current.json"
    previous = tmp_path / "previous.json"
    ratios = tmp_path / "ratios.json"
    summary = tmp_path / "summary.md"
    _write_benchmarks(
        current,
        {
            "test_sequential_fuse_284": 1.8,
            "test_batched_fuse_284": 0.02,
        },
    )
    _write_benchmarks(
        previous,
        {
            "test_sequential_fuse_284": 1.7,
            "test_batched_fuse_284": 0.015,
        },
    )
    ratios.write_text(
        json.dumps(
            {
                "same_run_ratios": [
                    {
                        "slower": "test_sequential_fuse_284",
                        "faster": "test_batched_fuse_284",
                        "min_ratio": 10.0,
                    }
                ],
                "previous_run": {"max_slowdown": 2.0},
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "--current",
                str(current),
                "--ratios",
                str(ratios),
                "--previous",
                str(previous),
                "--summary",
                str(summary),
            ]
        )
        == 0
    )
    assert "Benchmark comparison" in summary.read_text(encoding="utf-8")
