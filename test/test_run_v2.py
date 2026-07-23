"""Tests for the persistent, test-blind Variable 2 runner."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import datasets
from bert_cpu import engine as cpu
from exercises import task_binary_classification as adult
from experiments import run_v1, run_v2


def _parameters(model: adult.AdultMLP) -> list[np.ndarray]:
    return [parameter.data.copy() for parameter in model.parameters()]


def test_frozen_v2_config_and_expected_costs():
    config = run_v2.load_config()
    run_v2.validate_frozen_config(config)

    assert [item["id"] for item in config["configurations"]] == [
        "S-BETA-0.5",
        "S-BETA-1",
        "S-BETA-2",
        "S-BETA-5",
    ]
    assert {item["expected_flops_per_epoch"] for item in config["configurations"]} == {
        850_711_121
    }
    assert {
        item["expected_inference_flops_total"]
        for item in config["configurations"]
    } == {94_632_384}

    changed_beta = copy.deepcopy(config)
    changed_beta["configurations"][0]["beta"] = 0.25
    with pytest.raises(ValueError, match="beta map"):
        run_v2.validate_frozen_config(changed_beta)

    changed_test_policy = copy.deepcopy(config)
    changed_test_policy["evaluation"]["test_policy"] = "always"
    with pytest.raises(ValueError, match="evaluation"):
        run_v2.validate_frozen_config(changed_test_policy)


def test_v2_schema_and_run_ids_keep_beta_explicit():
    assert run_v2.RESULT_FIELDS[5] == "beta"
    assert run_v2.expected_run_id("S-BETA-2", 1) == "S-BETA-2-s1-r1"
    assert (
        run_v2.expected_run_id("S-BETA-2", 1, smoke=True)
        == "SMOKE-S-BETA-2-s1-r1"
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("beta", 2.0),
        ("run_id", "S-BETA-1-s2-r1"),
        ("flops_per_epoch", 1),
        ("flops_total", 1),
        ("inference_flops_total", 1),
        ("status", "failed"),
        ("test_accuracy", 0.9),
        ("notes", "official_test_loaded"),
    ),
)
def test_artifact_validation_rejects_frozen_identity_drift_before_files(
    field,
    invalid,
    monkeypatch,
):
    config = run_v2.load_config()
    selected = run_v2.get_configuration(config, "S-BETA-1")
    epoch_flops = selected["expected_flops_per_epoch"]
    total_flops = epoch_flops * config["training"]["epochs"]
    inference_flops = selected["expected_inference_flops_total"]
    row = {
        "run_id": "S-BETA-1-s0-r1",
        "task": config["task"],
        "variable": config["variable"],
        "config_id": selected["id"],
        "activation": selected["activation"],
        "beta": selected["beta"],
        "seed": 0,
        "repetition": 1,
        "run_kind": "scientific",
        "phase": config["phase"],
        "purpose": "primary",
        "status": "completed_valid",
        "config_hash": run_v1.sha256_file(run_v2.DEFAULT_CONFIG),
        "dataset_hash": config["data"]["encoded_train_sha256"],
        "split_seed": config["training"]["split_seed"],
        "split_hash": config["data"]["split_sha256"],
        "initial_weights_hash": config["data"]["initial_weights_sha256"]["0"],
        "epochs": config["training"]["epochs"],
        "train_samples": 26_049,
        "val_samples": 6_512,
        "parameters": config["architecture"]["parameters"],
        "test_accuracy": "",
        "flops_per_epoch": epoch_flops,
        "flops_total": total_flops,
        "gflops_total": total_flops / 1e9,
        "inference_flops_total": inference_flops,
        "inference_flops_per_sample": inference_flops / 6_512,
        "notes": "official_test_not_loaded",
    }
    row[field] = invalid
    monkeypatch.setattr(
        run_v2.common,
        "_result_artifacts_valid",
        lambda candidate: pytest.fail("invalid identity reached file validation"),
    )

    assert not run_v2.result_artifacts_valid(row)


@pytest.mark.parametrize("beta", (0.5, 1.0, 2.0, 5.0))
def test_v2_diagnostics_match_formula_and_do_not_mutate_state(beta):
    model = adult.build_model(
        2,
        activation="softplus_beta",
        model_seed=3,
        activation_beta=beta,
    )
    model.fc1.weight.data[...] = 0.0
    model.fc1.weight.data[0] = np.linspace(-2.0, 2.0, 64)
    model.fc1.weight.data[1] = np.linspace(0.25, -0.25, 64)
    X = np.array([[-2.0, 0.0, 3.0], [1.0, -1.0, 0.5]])
    weights_before = _parameters(model)
    rng_before = np.random.get_state()
    cpu.reset_flops()

    diagnostic = run_v2.activation_diagnostics(model, X, epoch=25)

    z = model.fc1.weight.data[1:].T @ X + model.fc1.weight.data[0][:, None]
    h = np.logaddexp(0.0, beta * z) / beta
    derivative = run_v1._stable_sigmoid(beta * z)
    assert diagnostic["beta"] == beta
    assert diagnostic["samples"] == 3
    assert diagnostic["values"] == 64 * 3
    assert diagnostic["h"]["mean"] == pytest.approx(float(h.mean()))
    assert diagnostic["local_derivative"]["mean"] == pytest.approx(
        float(derivative.mean())
    )
    assert (
        diagnostic["low_derivative_fraction"]
        + diagnostic["transition_derivative_fraction"]
        + diagnostic["high_derivative_fraction"]
        == pytest.approx(1.0)
    )
    assert cpu.flop_count() == 0
    for expected, parameter in zip(weights_before, model.parameters()):
        np.testing.assert_array_equal(parameter.data, expected)
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]


def test_results_csv_is_atomic_beta_aware_and_rejects_duplicate(tmp_path):
    path = tmp_path / "results.csv"
    row = {field: "" for field in run_v2.RESULT_FIELDS}
    row.update({"run_id": "S-BETA-1-s0-r1", "beta": 1.0})

    run_v2._append_result(path, row)
    with pytest.raises(FileExistsError):
        run_v2._append_result(path, row)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == run_v2.RESULT_FIELDS
    assert rows[0]["run_id"] == "S-BETA-1-s0-r1"
    assert rows[0]["beta"] == "1.0"


def test_scientific_reference_rejects_dirty_source_before_loading_data(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(run_v2, "_source_status", lambda: [" M source.py"])
    monkeypatch.setattr(
        run_v2.datasets,
        "load_adult",
        lambda split: pytest.fail("dirty scientific source must not load data"),
    )

    with pytest.raises(RuntimeError, match="clean source"):
        run_v2.execute_run(
            config_id="S-BETA-1",
            seed=0,
            artifacts_dir=tmp_path,
            verbose=False,
        )


def test_variant_is_blocked_before_three_reference_seeds(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_v2.datasets,
        "load_adult",
        lambda split: pytest.fail("variant must be blocked before loading data"),
    )

    with pytest.raises(RuntimeError, match="S-BETA-1"):
        run_v2.execute_run(
            config_id="S-BETA-2",
            seed=0,
            artifacts_dir=tmp_path,
            verbose=False,
        )


def test_reference_gate_requires_exactly_three_valid_seeds(monkeypatch):
    rows = []
    for seed in (0, 1, 2):
        row = {field: "same" for field in run_v2.RESULT_FIELDS}
        row.update(
            {
                "config_id": "S-BETA-1",
                "seed": str(seed),
                "repetition": "1",
                "status": "completed_valid",
            }
        )
        rows.append(row)

    monkeypatch.setattr(
        run_v2,
        "result_artifacts_valid",
        lambda row, **kwargs: True,
    )
    assert run_v2._reference_complete(rows, verify_artifacts=True)
    rows[-1]["seed"] = "1"
    assert not run_v2._reference_complete(rows, verify_artifacts=True)


def test_end_to_end_v2_smoke_is_isolated_and_never_loads_test(
    monkeypatch,
    tmp_path,
):
    original_load = datasets.load_adult
    loaded_splits: list[str] = []

    def tracked_load(split: str):
        loaded_splits.append(split)
        assert split == "train"
        return original_load(split)

    monkeypatch.setattr(run_v2.datasets, "load_adult", tracked_load)
    row = run_v2.execute_run(
        config_id="S-BETA-2",
        seed=0,
        smoke=True,
        artifacts_dir=tmp_path,
        verbose=False,
        command=["test-v2-smoke"],
    )

    assert loaded_splits == ["train"]
    assert row["run_kind"] == "smoke"
    assert row["status"] == "smoke_passed"
    assert row["beta"] == 2.0
    assert row["epochs"] == 2
    assert row["test_accuracy"] == ""
    assert row["flops_per_epoch"] == 850_711_121
    assert row["inference_flops_total"] == 94_632_384
    assert not (tmp_path / "results.csv").exists()

    log_path = Path(row["log_path"])
    checkpoint_path = Path(row["checkpoint_path"])
    assert log_path.exists()
    assert checkpoint_path.exists()
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert events[0]["event"] == "run_attempt_started"
    started = next(event for event in events if event["event"] == "run_started")
    assert started["data"]["official_test_loaded"] is False
    assert started["beta"] == 2.0
    assert [event["epoch"] for event in events if event["event"] == "epoch"] == [1, 2]
    assert [
        event["epoch"] for event in events if event["event"] == "diagnostic"
    ] == [0, 1]
    assert events[-1]["event"] == "run_completed"

    serialized = {field: str(row.get(field, "")) for field in run_v2.RESULT_FIELDS}
    assert run_v1._result_artifacts_valid(serialized)
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["metadata_json"].item()))
    assert manifest["beta"] == 2.0
    assert manifest["variable"] == "V2_softplus_curvature"
