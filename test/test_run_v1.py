"""Tests for the persistent, test-blind Variable 1 runner."""

from __future__ import annotations

import csv
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import datasets
from bert_cpu import engine as cpu
from experiments import run_v1
from exercises import task_binary_classification as adult


def _parameters(model: adult.AdultMLP) -> list[np.ndarray]:
    return [parameter.data.copy() for parameter in model.parameters()]


def test_published_split_and_initial_weight_hashes_are_reproduced():
    train = datasets.load_adult("train")
    train_indices, val_indices = adult.train_val_indices(train.n_samples, split_seed=0)
    assert run_v1.hash_arrays(train_indices, val_indices) == (
        "118bb0951fa1e0c4d88a4bbb493635d132492e2a52fb94698b792a99cec47bc0"
    )

    expected = {
        0: "9b1fe0ce77aacdb4ec92847cb6026b61ec028f18a3948ee85873448d664f4596",
        1: "d34dd1e6783a999faff6e235aefd23cd1e85a06bbe126423aec25bb1d8f867ae",
        2: "0310d0d1c0c02d2fc0754b90f2d9b35150972361d64c9333bf52541163c26a29",
    }
    for seed, digest in expected.items():
        model = adult.build_model(train.n_features, "relu", model_seed=seed)
        assert run_v1.parameter_hash(model) == digest

    reference = run_v1.parameter_hash(adult.build_model(108, "relu", model_seed=0))
    for activation in adult.ACTIVATIONS[1:]:
        assert run_v1.parameter_hash(adult.build_model(108, activation, 0)) == reference


def test_frozen_config_rejects_activation_or_evaluation_drift():
    config = run_v1.load_config()
    run_v1.validate_frozen_config(config)

    changed_activation = copy.deepcopy(config)
    changed_activation["configurations"][0]["activation"] = "swish"
    with pytest.raises(ValueError, match="activation map"):
        run_v1.validate_frozen_config(changed_activation)

    changed_policy = copy.deepcopy(config)
    changed_policy["evaluation"]["test_policy"] = "always"
    with pytest.raises(ValueError, match="evaluation"):
        run_v1.validate_frozen_config(changed_policy)


@pytest.mark.parametrize("activation", adult.ACTIVATIONS)
def test_diagnostics_are_finite_and_do_not_mutate_state(activation):
    model = adult.build_model(2, activation, model_seed=3)
    model.fc1.weight.data[...] = 0.0
    model.fc1.weight.data[0] = np.linspace(-1000.0, 1000.0, 64)
    X = np.zeros((2, 5))
    X_before = X.copy()
    for index, parameter in enumerate(model.parameters(), start=1):
        parameter.grad[...] = index + 0.25

    weights_before = _parameters(model)
    gradients_before = [parameter.grad.copy() for parameter in model.parameters()]
    rng_before = np.random.get_state()
    cpu.reset_flops()
    flops_before = cpu.flop_count()

    diagnostic = run_v1.activation_diagnostics(model, X, epoch=0)

    assert diagnostic["population"] == "validation"
    assert diagnostic["samples"] == 5
    assert diagnostic["values"] == 64 * 5
    assert set(diagnostic["z"]) == {
        "mean", "std", "min", "max", "p01", "p05", "p50", "p95", "p99"
    }
    assert np.isfinite(list(diagnostic["z"].values())).all()
    assert np.isfinite(list(diagnostic["h"].values())).all()
    assert np.isfinite(list(diagnostic["local_derivative"].values())).all()
    assert cpu.flop_count() == flops_before
    np.testing.assert_array_equal(X, X_before)
    for before, parameter in zip(weights_before, model.parameters()):
        np.testing.assert_array_equal(parameter.data, before)
    for before, parameter in zip(gradients_before, model.parameters()):
        np.testing.assert_array_equal(parameter.grad, before)
    after = np.random.get_state()
    assert rng_before[0] == after[0]
    np.testing.assert_array_equal(rng_before[1], after[1])
    assert rng_before[2:] == after[2:]


def _expected_summary(values: np.ndarray) -> dict[str, float]:
    result = {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    result.update(
        {
            f"p{percentile:02d}": float(np.percentile(values, percentile))
            for percentile in (1, 5, 50, 95, 99)
        }
    )
    return result


@pytest.mark.parametrize("activation", adult.ACTIVATIONS)
def test_diagnostic_values_derivatives_and_percentiles(activation):
    model = adult.build_model(2, activation, model_seed=0)
    model.fc1.weight.data[...] = 0.0
    model.fc1.weight.data[0] = np.linspace(-1.5, 1.5, 64)
    model.fc1.weight.data[1] = np.linspace(0.25, -0.25, 64)
    model.fc1.weight.data[2] = 0.1
    X = np.array([[-2.0, 0.0, 3.0], [1.0, -1.0, 0.5]])

    z = model.fc1.weight.data[1:].T @ X + model.fc1.weight.data[0][:, None]
    sigmoid = np.empty_like(z)
    positive = z >= 0.0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    sigmoid[~positive] = exp_z / (1.0 + exp_z)
    if activation == "relu":
        h = np.maximum(0.0, z)
        derivative = (z > 0.0).astype(z.dtype)
    elif activation == "sigmoid":
        h = sigmoid
        derivative = sigmoid * (1.0 - sigmoid)
    elif activation == "swish":
        h = z * sigmoid
        derivative = sigmoid + z * sigmoid * (1.0 - sigmoid)
    else:
        h = np.logaddexp(0.0, z)
        derivative = sigmoid

    diagnostic = run_v1.activation_diagnostics(model, X, epoch=25)
    for key, expected in (
        ("z", z),
        ("h", h),
        ("local_derivative", derivative),
    ):
        actual_summary = diagnostic[key]
        expected_summary = _expected_summary(expected)
        assert actual_summary.keys() == expected_summary.keys()
        for statistic, expected_value in expected_summary.items():
            assert np.isclose(actual_summary[statistic], expected_value, atol=1e-12)


def test_full_diagnostic_schedule_is_frozen():
    scheduled = [0] + [
        epoch for epoch in range(1, 101) if epoch in run_v1.DIAGNOSTIC_EPOCHS
    ]
    assert scheduled == [0, 1, 25, 50, 75, 100]


def test_relu_and_sigmoid_specific_diagnostic_fractions():
    X = np.zeros((1, 2))

    relu = adult.build_model(1, "relu", model_seed=0)
    relu.fc1.weight.data[...] = 0.0
    relu.fc1.weight.data[0, :16] = -1.0
    relu.fc1.weight.data[0, 32:] = 1.0
    relu_diagnostic = run_v1.activation_diagnostics(relu, X, epoch=0)
    assert relu_diagnostic["exact_zero_h_fraction"] == 0.5
    assert relu_diagnostic["near_zero_h_fraction"] == 0.5

    sigmoid = adult.build_model(1, "sigmoid", model_seed=0)
    sigmoid.fc1.weight.data[...] = 0.0
    sigmoid.fc1.weight.data[0, :32] = -10.0
    sigmoid.fc1.weight.data[0, 32:] = 10.0
    sigmoid_diagnostic = run_v1.activation_diagnostics(sigmoid, X, epoch=0)
    assert sigmoid_diagnostic["sigmoid_saturation_fraction"] == 1.0


def test_diagnostic_callback_does_not_change_training():
    rng = np.random.RandomState(8)
    X = rng.normal(size=(4, 24))
    y = np.array([0, 1] * 12)
    X_train, y_train, X_val, y_val = adult.train_val_split(X, y, split_seed=0)

    plain = adult.build_model(4, "swish", model_seed=1)
    observed_epochs: list[int] = []
    observed = adult.build_model(4, "swish", model_seed=1)

    plain_result = adult.train(
        plain, X_train, y_train, X_val, y_val, epochs=2, verbose=False
    )

    def callback(metrics: adult.EpochMetrics, model: adult.AdultMLP) -> None:
        observed_epochs.append(metrics.epoch)
        run_v1.activation_diagnostics(model, X_val, epoch=metrics.epoch)

    observed_result = adult.train(
        observed,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=2,
        verbose=False,
        epoch_callback=callback,
    )

    assert observed_epochs == [1, 2]
    assert observed_result == plain_result
    for expected, actual in zip(plain.parameters(), observed.parameters()):
        np.testing.assert_array_equal(actual.data, expected.data)


def test_checkpoint_round_trip_without_pickle(tmp_path):
    model = adult.build_model(5, "softplus", model_seed=2)
    expected_hash = run_v1.parameter_hash(model)
    path = tmp_path / "checkpoint.npz"
    metadata = {
        "run_id": "test",
        "run_kind": "smoke",
        "epoch": 2,
        "activation": "softplus",
        "architecture": {
            "input_features": 5,
            "hidden_features": 64,
            "output_classes": 2,
        },
        "final_weights_hash": expected_hash,
    }

    digest = run_v1.save_checkpoint(path, model, metadata)
    model.fc1.weight.data[...] = 0.0
    manifest = run_v1.load_checkpoint(path, model, expected_sha256=digest)

    assert digest == run_v1.sha256_file(path)
    assert manifest["run_id"] == "test"
    assert manifest["parameter_names"] == ["fc1.weight", "fc2.weight"]
    assert run_v1.parameter_hash(model) == expected_hash

    incompatible = adult.build_model(5, "relu", model_seed=9)
    incompatible_before = _parameters(incompatible)
    with pytest.raises(ValueError, match="activation"):
        run_v1.load_checkpoint(path, incompatible, expected_sha256=digest)
    for expected, parameter in zip(incompatible_before, incompatible.parameters()):
        np.testing.assert_array_equal(parameter.data, expected)

    with pytest.raises(ValueError, match="SHA-256"):
        run_v1.load_checkpoint(path, model, expected_sha256="0" * 64)


def test_results_append_is_atomic_and_rejects_duplicate(tmp_path):
    path = tmp_path / "results.csv"
    row = {field: "" for field in run_v1.RESULT_FIELDS}
    row["run_id"] = "F-RELU-s0-r1"

    run_v1._append_result(path, row)
    with pytest.raises(FileExistsError):
        run_v1._append_result(path, row)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["run_id"] == row["run_id"]


def test_run_reservation_is_atomic_and_never_reused(tmp_path):
    reservation = run_v1._reserve_run(tmp_path, "F-RELU-s0-r1")
    assert reservation.exists()
    with pytest.raises(FileExistsError, match="reserved"):
        run_v1._reserve_run(tmp_path, "F-RELU-s0-r1")


def _reproduction_row(*, repetition: int, strings: bool = False) -> dict[str, object]:
    row: dict[str, object] = {field: "same" for field in run_v1.RESULT_FIELDS}
    row.update(
        {
            "config_id": "F-RELU",
            "seed": 0,
            "repetition": repetition,
            "status": "completed_valid",
            "parameters": 7106,
            "flops_per_epoch": 100,
            "flops_total": 10_000,
            "inference_flops_total": 200,
            "train_loss_final": 0.3,
            "val_loss_final": 0.4,
            "train_accuracy": 0.8,
            "val_accuracy": 0.79,
            "inference_flops_per_sample": 12.5,
        }
    )
    if strings:
        return {key: str(value) for key, value in row.items()}
    return row


def test_relu_reproduction_uses_final_weights_metrics_and_flops():
    reference = _reproduction_row(repetition=1, strings=True)
    repeated = _reproduction_row(repetition=2)

    run_v1.validate_relu_reproduction(repeated, [reference])
    repeated["final_weights_hash"] = "different"
    with pytest.raises(RuntimeError, match="final_weights_hash"):
        run_v1.validate_relu_reproduction(repeated, [reference])


def test_baseline_gate_accepts_any_valid_second_seed_zero_repetition():
    rows = [
        _reproduction_row(repetition=1, strings=True),
        _reproduction_row(repetition=3, strings=True),
    ]
    rows[1]["repetition"] = "3"
    for seed in (1, 2):
        row = _reproduction_row(repetition=1, strings=True)
        row["seed"] = str(seed)
        rows.append(row)
    assert run_v1._baseline_complete(rows)

    expected = {
        "commit": "same",
        "config_hash": "same",
        "split_hash": "same",
    }
    assert run_v1._baseline_matches_context(rows, expected)
    rows[-1]["commit"] = "different"
    assert not run_v1._baseline_matches_context(rows, expected)


def test_scientific_run_rejects_dirty_source_before_loading_data(monkeypatch, tmp_path):
    monkeypatch.setattr(run_v1, "_source_status", lambda: [" M source.py"])
    monkeypatch.setattr(
        run_v1.datasets,
        "load_adult",
        lambda split: pytest.fail("data must not load for dirty scientific source"),
    )

    with pytest.raises(RuntimeError, match="clean source"):
        run_v1.execute_run(
            config_id="F-RELU",
            seed=0,
            artifacts_dir=tmp_path,
            verbose=False,
        )


def test_variant_is_blocked_before_complete_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_v1.datasets,
        "load_adult",
        lambda split: pytest.fail("variant must be blocked before loading data"),
    )
    with pytest.raises(RuntimeError, match="baseline"):
        run_v1.execute_run(
            config_id="F-SWISH",
            seed=0,
            artifacts_dir=tmp_path,
            verbose=False,
        )


def test_failure_before_data_initialization_is_logged_and_retry_gets_new_id(
    monkeypatch, tmp_path
):
    def fail_load(split: str):
        raise RuntimeError("synthetic loader failure")

    monkeypatch.setattr(run_v1.datasets, "load_adult", fail_load)
    with pytest.raises(RuntimeError, match="synthetic"):
        run_v1.execute_run(
            config_id="F-RELU",
            seed=0,
            repetition=1,
            smoke=True,
            artifacts_dir=tmp_path,
            verbose=False,
        )

    first_log = tmp_path / "logs/smoke/SMOKE-F-RELU-s0-r1.jsonl"
    events = [json.loads(line) for line in first_log.read_text().splitlines()]
    assert events[0]["event"] == "run_attempt_started"
    assert events[-1]["event"] == "run_failed"
    assert events[-1]["error_type"] == "RuntimeError"

    with pytest.raises(FileExistsError):
        run_v1.execute_run(
            config_id="F-RELU",
            seed=0,
            repetition=1,
            smoke=True,
            artifacts_dir=tmp_path,
            verbose=False,
        )
    with pytest.raises(RuntimeError, match="synthetic"):
        run_v1.execute_run(
            config_id="F-RELU",
            seed=0,
            repetition=2,
            smoke=True,
            artifacts_dir=tmp_path,
            verbose=False,
        )
    assert (tmp_path / "logs/smoke/SMOKE-F-RELU-s0-r2.jsonl").exists()


def test_end_to_end_smoke_writes_isolated_artifacts_and_never_test(
    monkeypatch, tmp_path
):
    original_load = datasets.load_adult
    loaded_splits: list[str] = []

    def tracked_load(split: str):
        loaded_splits.append(split)
        assert split == "train"
        return original_load(split)

    monkeypatch.setattr(run_v1.datasets, "load_adult", tracked_load)
    row = run_v1.execute_run(
        config_id="F-RELU",
        seed=0,
        smoke=True,
        artifacts_dir=tmp_path,
        verbose=False,
        command=["test-smoke"],
    )

    assert loaded_splits == ["train"]
    assert row["run_kind"] == "smoke"
    assert row["status"] == "smoke_passed"
    assert row["epochs"] == 2
    assert row["test_accuracy"] == ""
    assert not (tmp_path / "results.csv").exists()

    log_path = Path(row["log_path"])
    checkpoint_path = Path(row["checkpoint_path"])
    assert log_path.exists()
    assert checkpoint_path.exists()
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert events[0]["event"] == "run_attempt_started"
    initialized = next(event for event in events if event["event"] == "run_started")
    assert initialized["data"]["official_test_loaded"] is False
    assert [event["epoch"] for event in events if event["event"] == "diagnostic"] == [0, 1]
    assert [event["epoch"] for event in events if event["event"] == "epoch"] == [1, 2]
    assert any(event["event"] == "artifacts_verified" for event in events)
    assert events[-1]["event"] == "run_completed"

    serialized_row = {field: str(row.get(field, "")) for field in run_v1.RESULT_FIELDS}
    assert run_v1._result_artifacts_valid(serialized_row)
    serialized_row["checkpoint_hash"] = "0" * 64
    assert not run_v1._result_artifacts_valid(serialized_row)
