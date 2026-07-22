"""Regression tests for the reproducible Variable 1 visual summary."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments import plot_v1


def test_v1_plot_loader_excludes_determinism_repeat():
    grouped = plot_v1.load_v1_runs()

    assert tuple(grouped) == plot_v1.CONFIG_ORDER
    assert sum(len(runs) for runs in grouped.values()) == 12
    for config_id, runs in grouped.items():
        assert [int(run.row["seed"]) for run in runs] == [0, 1, 2]
        assert all(run.row["purpose"] == "primary" for run in runs)
        assert all(len(run.epochs) == 100 for run in runs)
        assert all(
            [item["epoch"] for item in run.diagnostics] == [0, 1, 25, 50, 75, 100]
            for run in runs
        )


def test_v1_aggregate_reproduces_final_table_and_pareto():
    curves, summary = plot_v1.aggregate_runs(plot_v1.load_v1_runs())
    by_id = {item["config_id"]: item for item in summary}

    expected_validation = {
        "F-RELU": 85.04811629811629,
        "F-SIGMOID": 84.74610974610974,
        "F-SWISH": 85.13513513513513,
        "F-SOFTPLUS": 84.88431613431614,
    }
    for config_id, expected in expected_validation.items():
        assert curves[config_id]["val_accuracy_values"].shape == (3, 100)
        assert np.isfinite(curves[config_id]["val_accuracy_values"]).all()
        assert np.isclose(by_id[config_id]["mean_val_accuracy_pct"], expected)

    assert by_id["F-SWISH"]["accuracy_rank"] == 1
    assert by_id["F-RELU"]["accuracy_rank"] == 2
    assert by_id["F-RELU"]["pareto_status"] == "pareto"
    assert by_id["F-SWISH"]["pareto_status"] == "pareto"
    assert by_id["F-SIGMOID"]["pareto_status"] == "dominated"
    assert by_id["F-SOFTPLUS"]["pareto_status"] == "dominated"


def test_v1_generator_writes_summary_and_three_pngs(tmp_path: Path):
    summary_path = tmp_path / "summary.csv"
    plots_dir = tmp_path / "plots"

    summary = plot_v1.generate(summary_path=summary_path, plots_dir=plots_dir)

    assert [item["config_id"] for item in summary] == [
        "F-SWISH",
        "F-RELU",
        "F-SOFTPLUS",
        "F-SIGMOID",
    ]
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert "\r" not in summary_path.read_text(encoding="utf-8")

    expected_plots = {
        "v1_learning_curves.png",
        "v1_final_metrics_by_seed.png",
        "v1_accuracy_vs_flops.png",
    }
    assert {path.name for path in plots_dir.iterdir()} == expected_plots
    for name in expected_plots:
        data = (plots_dir / name).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 10_000
