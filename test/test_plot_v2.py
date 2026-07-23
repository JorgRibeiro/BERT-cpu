"""Tests for the reproducible Variable 2 table and plots.

The fixtures emulate already-recorded artifacts; no model training is run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import plot_v2, run_v2


FINAL_VALIDATION_PCT = {
    "S-BETA-0.5": (80.0, 80.1, 80.2),
    "S-BETA-1": (81.0, 80.9, 81.1),
    "S-BETA-2": (80.8, 80.85, 80.9),
    "S-BETA-5": (80.2, 80.3, 80.1),
}


def _epoch_event(epoch: int, final_validation_pct: float) -> dict[str, object]:
    progress = epoch / 100.0
    val_accuracy = final_validation_pct / 100.0 - 0.02 * (1.0 - progress)
    if epoch == 75:
        val_accuracy += 0.01
    train_accuracy = val_accuracy + 0.02
    return {
        "event": "epoch",
        "epoch": epoch,
        "train_loss": 0.7 - 0.35 * progress,
        "val_loss": 0.72 - 0.32 * progress,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "flops": 850_711_121,
    }


def _write_v2_fixture(tmp_path: Path) -> Path:
    rows: list[dict[str, str]] = []
    for config_id in plot_v2.CONFIG_ORDER:
        for seed, final_validation_pct in enumerate(
            FINAL_VALIDATION_PCT[config_id]
        ):
            run_id = run_v2.expected_run_id(config_id, seed)
            log_path = tmp_path / "logs" / f"{run_id}.jsonl"
            checkpoint_path = tmp_path / "checkpoints" / f"{run_id}.npz"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"fixture checkpoint")

            events: list[dict[str, object]] = [
                {
                    "event": "run_started",
                    "data": {
                        "official_test_loaded": False,
                        "validation_majority_accuracy": 0.75,
                    },
                },
                {"event": "diagnostic", "epoch": 0},
            ]
            for epoch in range(1, 101):
                events.append(_epoch_event(epoch, final_validation_pct))
                if epoch in plot_v2.DIAGNOSTIC_EPOCHS[1:]:
                    events.append({"event": "diagnostic", "epoch": epoch})
            events.append(
                {
                    "event": "run_completed",
                    "status": "completed_valid",
                    "result_registered": True,
                }
            )
            log_path.write_text(
                "".join(
                    json.dumps(event, sort_keys=True) + "\n" for event in events
                ),
                encoding="utf-8",
            )
            final_event = next(
                event
                for event in reversed(events)
                if event["event"] == "epoch"
            )

            row = {field: "" for field in run_v2.RESULT_FIELDS}
            row.update(
                {
                    "run_id": run_id,
                    "task": "adult_binary_classification",
                    "variable": "V2_softplus_curvature",
                    "config_id": config_id,
                    "activation": "softplus_beta",
                    "beta": str(plot_v2.BETAS[config_id]),
                    "seed": str(seed),
                    "repetition": "1",
                    "run_kind": "scientific",
                    "phase": "v2_train_validation",
                    "purpose": "primary",
                    "status": "completed_valid",
                    "branch": "q01-ativacoes-adult",
                    "commit": "fixture-commit",
                    "base_commit": "fixture-base",
                    "code_state_hash": "fixture-source",
                    "config_hash": "fixture-config",
                    "environment_hash": "fixture-environment",
                    "dataset_hash": "fixture-dataset",
                    "split_seed": "0",
                    "split_hash": "fixture-split",
                    "initial_weights_hash": f"fixture-initial-seed-{seed}",
                    "final_weights_hash": f"fixture-final-{run_id}",
                    "epochs": "100",
                    "optimizer": "Adam",
                    "learning_rate": "0.01",
                    "train_samples": "26049",
                    "val_samples": "6512",
                    "parameters": "7106",
                    "train_loss_final": str(final_event["train_loss"]),
                    "val_loss_final": str(final_event["val_loss"]),
                    "train_accuracy": str(final_event["train_accuracy"]),
                    "val_accuracy": str(final_event["val_accuracy"]),
                    "test_accuracy": "",
                    "flops_per_epoch": "850711121",
                    "flops_total": "85071112100",
                    "gflops_total": "85.0711121",
                    "inference_flops_total": "94632384",
                    "inference_flops_per_sample": "14532",
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_hash": "fixture-checkpoint-hash",
                    "log_path": str(log_path),
                    "notes": "synthetic fixture; no training",
                }
            )
            rows.append(row)

    results_path = tmp_path / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=run_v2.RESULT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return results_path


@pytest.fixture
def recorded_v2_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    results_path = _write_v2_fixture(tmp_path)

    def fixture_artifacts_are_valid(row: dict[str, str]) -> bool:
        return (
            Path(row["log_path"]).is_file()
            and Path(row["checkpoint_path"]).is_file()
        )

    monkeypatch.setattr(
        run_v2,
        "result_artifacts_valid",
        fixture_artifacts_are_valid,
    )
    return results_path


def test_v2_loader_requires_complete_grid_and_empty_test(
    recorded_v2_fixture: Path,
):
    grouped = plot_v2.load_v2_runs(recorded_v2_fixture)

    assert tuple(grouped) == plot_v2.CONFIG_ORDER
    assert sum(len(runs) for runs in grouped.values()) == 12
    for config_id, runs in grouped.items():
        assert [int(run.row["seed"]) for run in runs] == [0, 1, 2]
        assert all(float(run.row["beta"]) == plot_v2.BETAS[config_id] for run in runs)
        assert all(run.row["test_accuracy"] == "" for run in runs)
        assert all(len(run.epochs) == 100 for run in runs)
        assert all(
            [item["epoch"] for item in run.diagnostics]
            == list(plot_v2.DIAGNOSTIC_EPOCHS)
            for run in runs
        )

    rows = run_v2.load_results(recorded_v2_fixture)
    rows[0]["test_accuracy"] = "0.99"
    with pytest.raises(ValueError, match="official-test accuracy"):
        plot_v2._load_recorded_run(rows[0])

    rows[0]["test_accuracy"] = ""
    rows[-1]["commit"] = "mixed-context-commit"
    mixed_path = recorded_v2_fixture.with_name("mixed_context.csv")
    with mixed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=run_v2.RESULT_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="experimental contexts: commit"):
        plot_v2.load_v2_runs(mixed_path)


def test_h2_uses_epoch_100_paired_seeds_and_exact_tie_policy():
    sustained = plot_v2.evaluate_h2(FINAL_VALIDATION_PCT)

    assert sustained.central_winners == ("S-BETA-1",)
    assert sustained.extreme_winners == ("S-BETA-5",)
    assert sustained.status == "sustained"
    assert len(sustained.comparisons) == 1
    comparison = sustained.comparisons[0]
    assert np.isclose(comparison.mean_difference_pp, 0.8)
    assert comparison.positive_seeds == 3
    assert comparison.negative_seeds == 0

    central_tie_with_disagreement = {
        "S-BETA-0.5": (80.0, 80.0, 80.0),
        "S-BETA-1": (80.5, 80.5, 80.5),
        "S-BETA-2": (79.0, 79.0, 83.5),
        "S-BETA-5": (79.0, 79.0, 79.0),
    }
    tied = plot_v2.evaluate_h2(central_tie_with_disagreement)

    assert tied.central_winners == ("S-BETA-1", "S-BETA-2")
    assert tied.extreme_winners == ("S-BETA-0.5",)
    assert [item.status for item in tied.comparisons] == [
        "sustained",
        "inconclusive",
    ]
    assert tied.status == "inconclusive"

    refuted = plot_v2.evaluate_h2(
        {
            "S-BETA-0.5": (81.0, 81.1, 80.9),
            "S-BETA-1": (80.0, 80.2, 80.1),
            "S-BETA-2": (80.2, 80.3, 80.1),
            "S-BETA-5": (80.5, 80.4, 80.6),
        }
    )
    assert refuted.central_winners == ("S-BETA-2",)
    assert refuted.extreme_winners == ("S-BETA-0.5",)
    assert refuted.status == "refuted"


def test_v2_aggregate_records_secondary_best_epochs(
    recorded_v2_fixture: Path,
):
    curves, summary, h2 = plot_v2.aggregate_runs(
        plot_v2.load_v2_runs(recorded_v2_fixture)
    )
    by_id = {item["config_id"]: item for item in summary}

    assert h2.status == "sustained"
    assert by_id["S-BETA-1"]["accuracy_rank"] == 1
    assert by_id["S-BETA-1"]["mean_val_accuracy_pct"] == pytest.approx(81.0)
    assert by_id["S-BETA-1"]["std_val_accuracy_pp"] == pytest.approx(0.1)
    for config_id in plot_v2.CONFIG_ORDER:
        assert curves[config_id]["val_accuracy_values"].shape == (3, 100)
        assert curves[config_id]["best_epochs"].tolist() == [75, 75, 75]
        assert by_id[config_id]["gflops_per_run"] == pytest.approx(85.0711121)
        assert (
            by_id[config_id]["mean_best_val_accuracy_pct"]
            > by_id[config_id]["mean_val_accuracy_pct"]
        )


def test_v2_generator_writes_summary_and_three_pngs(
    recorded_v2_fixture: Path,
    tmp_path: Path,
):
    summary_path = tmp_path / "generated" / "summary.csv"
    plots_dir = tmp_path / "generated" / "plots"

    summary, h2 = plot_v2.generate(
        results_path=recorded_v2_fixture,
        summary_path=summary_path,
        plots_dir=plots_dir,
    )

    assert h2.status == "sustained"
    assert summary[0]["config_id"] == "S-BETA-1"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert "\r" not in summary_path.read_text(encoding="utf-8")
    assert {row["h2_status"] for row in rows} == {"sustained"}

    expected_plots = {
        "learning_curves.png",
        "final_metrics_by_seed.png",
        "validation_vs_beta.png",
    }
    assert {path.name for path in plots_dir.iterdir()} == expected_plots
    for name in expected_plots:
        data = (plots_dir / name).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 10_000
