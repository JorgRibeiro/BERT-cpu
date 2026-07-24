"""Tests for the pre-registered Variable 3 analysis rules and plots."""

from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pytest

from experiments import plot_v3


def test_h3_rules_use_paired_seeds_threshold_cost_and_return_order():
    final = {
        "L1-DIRECT": (80.0, 80.1, 79.9),
        "L2-IDENTITY": (80.1, 80.2, 80.0),
        "L3-IDENTITY": (79.8, 80.0, 79.9),
    }
    evaluation = plot_v3.evaluate_h3(
        final,
        relu_validation_pct=(80.8, 80.9, 80.7),
        parameters={
            "L1-DIRECT": 218,
            "L2-IDENTITY": 7_106,
            "L3-IDENTITY": 11_266,
        },
        gflops={
            "L1-DIRECT": 2.6107501,
            "L2-IDENTITY": 84.0291601,
            "L3-IDENTITY": 154.4654481,
        },
        returns={
            "L1-DIRECT": 2.0,
            "L2-IDENTITY": 0.07,
            "L3-IDENTITY": 0.03,
        },
    )

    assert evaluation.h3a_status == "not_contradicted"
    assert [item.status for item in evaluation.h3a_comparisons] == [
        "not_contradicted",
        "not_contradicted",
    ]
    assert evaluation.h3b_status == "sustained"
    assert evaluation.h3c_status == "sustained"
    assert evaluation.h3d_status == "sustained"
    assert evaluation.relu_minus_l2_positive_seeds == 3


def test_h3a_contradiction_h3c_reversal_and_h3d_refutation():
    evaluation = plot_v3.evaluate_h3(
        {
            "L1-DIRECT": (80.0, 80.0, 80.0),
            "L2-IDENTITY": (80.6, 80.7, 80.5),
            "L3-IDENTITY": (80.1, 80.0, 80.2),
        },
        relu_validation_pct=(79.9, 79.8, 79.7),
        parameters={
            "L1-DIRECT": 218,
            "L2-IDENTITY": 7_106,
            "L3-IDENTITY": 11_266,
        },
        gflops={
            "L1-DIRECT": 2.6,
            "L2-IDENTITY": 84.0,
            "L3-IDENTITY": 154.0,
        },
        returns={
            "L1-DIRECT": 0.01,
            "L2-IDENTITY": 0.02,
            "L3-IDENTITY": 0.03,
        },
    )

    assert evaluation.h3a_status == "contradicted"
    assert evaluation.h3a_comparisons[0].status == "contradicted"
    assert evaluation.h3c_status == "refuted"
    assert evaluation.h3d_status == "refuted"


def test_directional_threshold_requires_mean_delta_and_two_signs():
    mean, positive, negative, status = plot_v3._directional_status(
        np.array([0.75, 0.75, 0.0])
    )
    assert mean == pytest.approx(0.5)
    assert (positive, negative, status) == (2, 0, "sustained")

    mean, positive, negative, status = plot_v3._directional_status(
        np.array([-0.75, -0.75, 0.0])
    )
    assert mean == pytest.approx(-0.5)
    assert (positive, negative, status) == (0, 2, "refuted")

    assert plot_v3._directional_status(
        np.array([1.0, -0.2, -0.2])
    )[3] == "inconclusive"


def test_h3c_tie_or_mixed_order_is_inconclusive():
    evaluation = plot_v3.evaluate_h3(
        {
            "L1-DIRECT": (80.0, 80.0, 80.0),
            "L2-IDENTITY": (80.1, 80.1, 80.1),
            "L3-IDENTITY": (80.2, 80.2, 80.2),
        },
        relu_validation_pct=(80.0, 80.0, 80.0),
        parameters={
            "L1-DIRECT": 218,
            "L2-IDENTITY": 7_106,
            "L3-IDENTITY": 11_266,
        },
        gflops={
            "L1-DIRECT": 2.6,
            "L2-IDENTITY": 84.0,
            "L3-IDENTITY": 154.0,
        },
        returns={
            "L1-DIRECT": 1.0,
            "L2-IDENTITY": 1.0,
            "L3-IDENTITY": 0.5,
        },
    )
    assert evaluation.h3c_status == "inconclusive"


def test_h3a_exact_threshold_and_h3b_bad_order():
    evaluation = plot_v3.evaluate_h3(
        {
            "L1-DIRECT": (80.0, 80.0, 80.0),
            "L2-IDENTITY": (80.75, 80.75, 80.0),
            "L3-IDENTITY": (80.0, 80.0, 80.0),
        },
        relu_validation_pct=(80.0, 80.0, 80.0),
        parameters={
            "L1-DIRECT": 218,
            "L2-IDENTITY": 100,
            "L3-IDENTITY": 11_266,
        },
        gflops={
            "L1-DIRECT": 2.6,
            "L2-IDENTITY": 84.0,
            "L3-IDENTITY": 154.0,
        },
        returns={
            "L1-DIRECT": 1.0,
            "L2-IDENTITY": 0.5,
            "L3-IDENTITY": 0.2,
        },
    )

    comparison = evaluation.h3a_comparisons[0]
    assert comparison.mean_gain_pp == pytest.approx(0.5)
    assert comparison.positive_seeds == 2
    assert comparison.status == "contradicted"
    assert evaluation.h3a_status == "contradicted"
    assert evaluation.h3b_status == "refuted"


def test_relu_bridge_checks_controls_initial_weights_and_relu_cost(tmp_path):
    l2_runs = []
    relu_runs = []
    common_fields = {
        "dataset_hash": "data",
        "split_seed": "0",
        "split_hash": "split",
        "epochs": "100",
        "optimizer": "Adam",
        "learning_rate": "0.01",
        "train_samples": "26049",
        "val_samples": "6512",
        "parameters": "7106",
    }
    data = {
        "raw_train_hash": "raw",
        "encoded_train_hash": "data",
        "features": 108,
        "samples": 32_561,
        "train_samples": 26_049,
        "validation_samples": 6_512,
        "split_seed": 0,
        "split_hash": "split",
        "validation_majority_accuracy": 0.755,
        "preprocessing_limitation": "encoder_fit_on_official_train_before_holdout",
        "official_test_loaded": False,
    }
    training = {
        "epochs": 100,
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "batching": "full_batch",
        "train_loss_timing": "before_adam_step",
        "validation_loss_timing": "after_adam_step",
        "accuracy_timing": "after_adam_step_outside_flop_window",
        "evaluate_test": False,
    }
    environment = {
        "python": "3.11",
        "numpy": "2.1",
        "default_dtype": "float64",
        "requirements_hash": "requirements",
        "platform": "platform-v1",
    }
    for seed in range(3):
        initial_hash = f"seed-{seed}"
        l2_log = tmp_path / f"l2-{seed}.jsonl"
        relu_log = tmp_path / f"relu-{seed}.jsonl"
        l2_started = {
            "event": "run_started",
            "data": data,
            "training": training,
            "environment": {**environment, "platform": "platform-v3"},
            "model": {
                "architecture": {
                    "input_features": 108,
                    "hidden_features": 64,
                    "output_classes": 2,
                    "parameters": 7_106,
                    "depth": 2,
                    "layer_sizes": [108, 64, 2],
                },
                "identity_operation_created": False,
            },
        }
        relu_started = {
            "event": "run_started",
            "data": data,
            "training": training,
            "environment": environment,
            "model": {
                "architecture": {
                    "input_features": 108,
                    "hidden_features": 64,
                    "output_classes": 2,
                    "parameters": 7_106,
                }
            },
        }
        l2_log.write_text(json.dumps(l2_started) + "\n", encoding="utf-8")
        relu_log.write_text(json.dumps(relu_started) + "\n", encoding="utf-8")
        l2_runs.append(
            SimpleNamespace(
                row={
                    **common_fields,
                    "seed": str(seed),
                    "config_id": "L2-IDENTITY",
                    "activation": "none",
                    "depth": "2",
                    "layer_sizes": "[108,64,2]",
                    "initial_weights_hash": initial_hash,
                    "flops_per_epoch": "840291601",
                    "inference_flops_total": "92548544",
                    "log_path": str(l2_log),
                },
                majority_accuracy=0.755,
            )
        )
        relu_runs.append(
            SimpleNamespace(
                row={
                    **common_fields,
                    "seed": str(seed),
                    "config_id": "F-RELU",
                    "activation": "relu",
                    "initial_weights_hash": initial_hash,
                    "flops_per_epoch": "842375505",
                    "inference_flops_total": "92965312",
                    "log_path": str(relu_log),
                },
                majority_accuracy=0.755,
            )
        )

    same_platform = plot_v3.validate_relu_bridge(
        {
            "L1-DIRECT": (),
            "L2-IDENTITY": tuple(l2_runs),
            "L3-IDENTITY": (),
        },
        tuple(relu_runs),
    )
    assert same_platform is False

    relu_runs[0].row["initial_weights_hash"] = "changed"
    with pytest.raises(ValueError, match="initial weights"):
        plot_v3.validate_relu_bridge(
            {
                "L1-DIRECT": (),
                "L2-IDENTITY": tuple(l2_runs),
                "L3-IDENTITY": (),
            },
            tuple(relu_runs),
        )


def test_aggregate_preserves_losses_best_epochs_return_rank_and_v3_pareto(
    monkeypatch,
):
    monkeypatch.setattr(
        plot_v3,
        "validate_relu_bridge",
        lambda grouped, relu_runs: False,
    )
    final_values = {
        "L1-DIRECT": (0.800, 0.801, 0.799),
        "L2-IDENTITY": (0.795, 0.796, 0.794),
        "L3-IDENTITY": (0.810, 0.811, 0.809),
    }
    parameters = {
        "L1-DIRECT": 218,
        "L2-IDENTITY": 7_106,
        "L3-IDENTITY": 11_266,
    }
    depths = {"L1-DIRECT": 1, "L2-IDENTITY": 2, "L3-IDENTITY": 3}
    flops = {
        "L1-DIRECT": 2_610_750_100,
        "L2-IDENTITY": 84_029_160_100,
        "L3-IDENTITY": 154_465_448_100,
    }
    grouped = {}
    for config_id in plot_v3.CONFIG_ORDER:
        runs = []
        for seed, final in enumerate(final_values[config_id]):
            epochs = []
            for epoch in range(1, 101):
                progress = epoch / 100
                val_accuracy = final - 0.02 * (1.0 - progress)
                if epoch == 75:
                    val_accuracy += 0.03
                epochs.append(
                    {
                        "epoch": epoch,
                        "train_loss": 0.7 - 0.35 * progress,
                        "val_loss": 0.72 - 0.32 * progress,
                        "train_accuracy": val_accuracy + 0.02,
                        "val_accuracy": val_accuracy,
                    }
                )
            runs.append(
                plot_v3.RecordedRun(
                    row={
                        "seed": str(seed),
                        "depth": str(depths[config_id]),
                        "parameters": str(parameters[config_id]),
                        "flops_total": str(flops[config_id]),
                    },
                    epochs=tuple(epochs),
                    diagnostics=(),
                    majority_accuracy=0.75,
                )
            )
        grouped[config_id] = tuple(runs)
    relu_runs = tuple(
        SimpleNamespace(epochs=({"val_accuracy": 0.82},))
        for _ in range(3)
    )

    curves, summary, h3 = plot_v3.aggregate_runs(grouped, relu_runs)
    by_id = {row["config_id"]: row for row in summary}

    assert by_id["L3-IDENTITY"]["accuracy_rank"] == 1
    assert by_id["L2-IDENTITY"]["pareto_status"] == "v3_dominated"
    assert by_id["L1-DIRECT"]["pareto_status"] == "v3_pareto"
    assert curves["L1-DIRECT"]["best_epochs"].tolist() == [75, 75, 75]
    assert by_id["L1-DIRECT"]["mean_train_loss"] == pytest.approx(0.35)
    assert by_id["L1-DIRECT"]["mean_val_loss"] == pytest.approx(0.40)
    expected_return = (
        by_id["L1-DIRECT"]["mean_val_accuracy_pct"] - 75.0
    ) / by_id["L1-DIRECT"]["gflops_per_run"]
    assert by_id["L1-DIRECT"]["return_pp_per_gflop"] == pytest.approx(
        expected_return
    )
    assert "h3a_l2_minus_l1_mean_pp" in by_id["L1-DIRECT"]
    assert h3.relu_bridge_same_platform is False


def test_three_v3_plots_are_valid_png_files(tmp_path):
    epochs = np.arange(1, 101)
    curves = {}
    for index, config_id in enumerate(plot_v3.CONFIG_ORDER):
        base = 0.78 + index * 0.002
        values = np.vstack(
            [
                base + 0.04 * (1.0 - np.exp(-epochs / 16)),
                base + 0.04 * (1.0 - np.exp(-epochs / 18)),
                base + 0.04 * (1.0 - np.exp(-epochs / 20)),
            ]
        )
        loss = np.vstack(
            [
                0.7 - 0.3 * (1.0 - np.exp(-epochs / 16)),
                0.7 - 0.3 * (1.0 - np.exp(-epochs / 18)),
                0.7 - 0.3 * (1.0 - np.exp(-epochs / 20)),
            ]
        )
        curves[config_id] = {
            "epochs": epochs,
            "train_loss_values": loss - 0.02,
            "train_loss_mean": (loss - 0.02).mean(axis=0),
            "train_loss_std": (loss - 0.02).std(axis=0, ddof=1),
            "val_loss_values": loss,
            "val_accuracy_values": values,
            "val_accuracy_mean": values.mean(axis=0),
            "val_accuracy_std": values.std(axis=0, ddof=1),
            "train_accuracy_values": values + 0.02,
            "train_accuracy_mean": (values + 0.02).mean(axis=0),
            "train_accuracy_std": (values + 0.02).std(axis=0, ddof=1),
            "val_loss_mean": loss.mean(axis=0),
            "val_loss_std": loss.std(axis=0, ddof=1),
        }

    summary = [
        {
            "config_id": config_id,
            "parameters": parameters,
            "gflops_per_run": gflops,
            "mean_val_accuracy_pct": float(
                curves[config_id]["val_accuracy_mean"][-1] * 100
            ),
            "std_val_accuracy_pp": float(
                curves[config_id]["val_accuracy_std"][-1] * 100
            ),
            "pareto_status": "v3_pareto",
        }
        for config_id, parameters, gflops in (
            ("L1-DIRECT", 218, 2.6107501),
            ("L2-IDENTITY", 7_106, 84.0291601),
            ("L3-IDENTITY", 11_266, 154.4654481),
        )
    ]
    relu_runs = tuple(
        SimpleNamespace(
            row={"flops_total": "84237550500"},
            epochs=({"val_accuracy": 0.85 + seed * 0.001},),
        )
        for seed in range(3)
    )

    paths = (
        tmp_path / "learning.png",
        tmp_path / "final.png",
        tmp_path / "tradeoff.png",
    )
    plot_v3.plot_learning_curves(paths[0], curves)
    plot_v3.plot_final_metrics(paths[1], curves)
    plot_v3.plot_accuracy_vs_flops(paths[2], summary, curves, relu_runs)

    for path in paths:
        data = path.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 10_000
