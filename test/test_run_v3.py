"""Tests for the persistent, test-blind Variable 3 unit runner."""

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
from experiments import run_v1, run_v3


def test_frozen_v3_config_ids_hashes_and_expected_costs():
    config = run_v3.load_config()
    run_v3.validate_frozen_config(config)

    assert [item["id"] for item in config["configurations"]] == list(
        run_v3.CONFIG_ORDER
    )
    assert [item["parameters"] for item in config["configurations"]] == [
        218,
        7_106,
        11_266,
    ]
    assert [
        item["expected_flops_per_epoch"] for item in config["configurations"]
    ] == [26_107_501, 840_291_601, 1_544_654_481]
    assert [
        item["expected_inference_flops_total"]
        for item in config["configurations"]
    ] == [2_839_232, 92_548_544, 146_728_384]

    changed_depth = copy.deepcopy(config)
    changed_depth["configurations"][1]["depth"] = 3
    with pytest.raises(ValueError, match="depth map"):
        run_v3.validate_frozen_config(changed_depth)

    changed_rule = copy.deepcopy(config)
    changed_rule["evaluation"]["decision_rules"]["H3a"] = "post_hoc"
    with pytest.raises(ValueError, match="evaluation"):
        run_v3.validate_frozen_config(changed_rule)


def test_v3_schema_and_run_ids_keep_depth_explicit():
    assert run_v3.RESULT_FIELDS[5:7] == ("depth", "layer_sizes")
    assert run_v3.expected_run_id("L2-IDENTITY", 1) == "L2-IDENTITY-s1-r1"
    assert (
        run_v3.expected_run_id("L2-IDENTITY", 1, smoke=True)
        == "SMOKE-L2-IDENTITY-s1-r1"
    )


@pytest.mark.parametrize("depth", (1, 2, 3))
def test_affine_diagnostics_match_model_without_mutating_state(depth):
    model = adult.build_linear_model(3, depth=depth, model_seed=4, hidden=5)
    X = np.linspace(-2.0, 2.0, 3 * 9).reshape(3, 9)
    weights_before = [parameter.data.copy() for parameter in model.parameters()]
    rng_before = np.random.get_state()
    cpu.reset_flops()

    diagnostic = run_v3.affine_diagnostics(model, X, epoch=25)

    assert diagnostic["depth"] == depth
    assert diagnostic["samples"] == 9
    assert diagnostic["equivalent_affine"] is True
    assert diagnostic["max_abs_error"] <= 1e-12
    assert diagnostic["collapsed_rank"] <= 2
    assert len(diagnostic["layers"]) == depth
    assert cpu.flop_count() == 0
    for expected, parameter in zip(weights_before, model.parameters()):
        np.testing.assert_array_equal(parameter.data, expected)
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]


def test_results_csv_is_atomic_depth_aware_and_rejects_duplicate(tmp_path):
    path = tmp_path / "results.csv"
    row = {field: "" for field in run_v3.RESULT_FIELDS}
    row.update(
        {
            "run_id": "L1-DIRECT-s0-r1",
            "depth": 1,
            "layer_sizes": "[108,2]",
        }
    )

    run_v3._append_result(path, row)
    with pytest.raises(FileExistsError):
        run_v3._append_result(path, row)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == run_v3.RESULT_FIELDS
    assert rows[0]["run_id"] == "L1-DIRECT-s0-r1"
    assert rows[0]["depth"] == "1"
    assert "\r" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("depth", 2),
        ("layer_sizes", "[108,64,2]"),
        ("activation", "identity"),
        ("initial_weights_hash", "changed"),
        ("flops_per_epoch", 1),
        ("flops_total", 1),
        ("inference_flops_total", 1),
        ("status", "failed"),
        ("test_accuracy", 0.9),
    ),
)
def test_artifact_validation_rejects_frozen_identity_drift_before_files(
    field,
    invalid,
    monkeypatch,
):
    config = run_v3.load_config()
    selected = run_v3.get_configuration(config, "L1-DIRECT")
    epoch_flops = selected["expected_flops_per_epoch"]
    total_flops = epoch_flops * config["training"]["epochs"]
    inference_flops = selected["expected_inference_flops_total"]
    row = {name: "" for name in run_v3.RESULT_FIELDS}
    row.update(
        {
            "run_id": "L1-DIRECT-s0-r1",
            "task": config["task"],
            "variable": config["variable"],
            "config_id": selected["id"],
            "activation": "none",
            "depth": 1,
            "layer_sizes": "[108,2]",
            "seed": 0,
            "repetition": 1,
            "run_kind": "scientific",
            "phase": config["phase"],
            "purpose": "primary",
            "status": "completed_valid",
            "config_hash": run_v1.sha256_file(run_v3.DEFAULT_CONFIG),
            "dataset_hash": config["data"]["encoded_train_sha256"],
            "split_seed": config["training"]["split_seed"],
            "split_hash": config["data"]["split_sha256"],
            "initial_weights_hash": (
                config["data"]["initial_weights_sha256"]["L1-DIRECT"]["0"]
            ),
            "epochs": config["training"]["epochs"],
            "train_samples": 26_049,
            "val_samples": 6_512,
            "parameters": selected["parameters"],
            "test_accuracy": "",
            "flops_per_epoch": epoch_flops,
            "flops_total": total_flops,
            "gflops_total": total_flops / 1e9,
            "inference_flops_total": inference_flops,
            "inference_flops_per_sample": inference_flops / 6_512,
            "notes": "official_test_not_loaded",
        }
    )
    row[field] = invalid
    monkeypatch.setattr(
        run_v3.common,
        "_result_artifacts_valid",
        lambda candidate: pytest.fail("invalid identity reached file validation"),
    )

    assert not run_v3.result_artifacts_valid(row)


def test_v3_checkpoint_round_trip_is_depth_specific(tmp_path):
    config = run_v3.load_config()
    selected = run_v3.get_configuration(config, "L3-IDENTITY")
    model = adult.build_linear_model(108, depth=3, model_seed=2)
    final_hash = run_v1.parameter_hash(model)
    manifest = {
        "run_id": "fixture",
        "activation": "none",
        "depth": 3,
        "layer_sizes": selected["layer_sizes"],
        "configuration": selected,
        "final_weights_hash": final_hash,
    }
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint_hash = run_v1.save_checkpoint(checkpoint, model, manifest)
    restored = adult.build_linear_model(108, depth=3, model_seed=0)

    loaded = run_v3._load_v3_checkpoint(
        checkpoint,
        restored,
        expected_sha256=checkpoint_hash,
        expected_manifest=manifest,
    )

    assert loaded["depth"] == 3
    assert run_v1.parameter_hash(restored) == final_hash
    wrong_depth = adult.build_linear_model(108, depth=2, model_seed=0)
    with pytest.raises(ValueError, match="names|shapes"):
        run_v3._load_v3_checkpoint(
            checkpoint,
            wrong_depth,
            expected_sha256=checkpoint_hash,
            expected_manifest=manifest,
        )


def test_scientific_reference_rejects_dirty_source_before_loading_data(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(run_v3, "_source_status", lambda: [" M source.py"])
    monkeypatch.setattr(
        run_v3.datasets,
        "load_adult",
        lambda split: pytest.fail("dirty scientific source must not load data"),
    )

    with pytest.raises(RuntimeError, match="clean source"):
        run_v3.execute_run(
            config_id="L1-DIRECT",
            seed=0,
            artifacts_dir=tmp_path,
            verbose=False,
        )


@pytest.mark.parametrize("config_id", ("L2-IDENTITY", "L3-IDENTITY"))
def test_deeper_model_is_blocked_before_predecessors(
    config_id,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        run_v3.datasets,
        "load_adult",
        lambda split: pytest.fail("blocked run must not load data"),
    )

    with pytest.raises(RuntimeError, match="blocked"):
        run_v3.execute_run(
            config_id=config_id,
            seed=0,
            artifacts_dir=tmp_path,
            verbose=False,
        )


def test_predecessor_gate_requires_three_valid_seeds(monkeypatch):
    rows = []
    for seed in (0, 1, 2):
        row = {field: "same" for field in run_v3.RESULT_FIELDS}
        row.update(
            {
                "config_id": "L1-DIRECT",
                "seed": str(seed),
                "repetition": "1",
                "status": "completed_valid",
            }
        )
        rows.append(row)

    monkeypatch.setattr(
        run_v3,
        "result_artifacts_valid",
        lambda row, **kwargs: True,
    )
    assert run_v3._configuration_complete(
        rows,
        "L1-DIRECT",
        verify_artifacts=True,
    )
    rows[-1]["seed"] = "1"
    assert not run_v3._configuration_complete(
        rows,
        "L1-DIRECT",
        verify_artifacts=True,
    )


def test_existing_l1_seed_from_another_commit_blocks_next_seed(monkeypatch):
    row = {field: "same" for field in run_v3.RESULT_FIELDS}
    row.update(
        {
            "run_id": "L1-DIRECT-s0-r1",
            "config_id": "L1-DIRECT",
            "seed": "0",
            "repetition": "1",
            "status": "completed_valid",
            "commit": "old-commit",
        }
    )
    expected = {
        "commit": "current-commit",
        "dataset_hash": "same",
    }
    monkeypatch.setattr(
        run_v3,
        "result_artifacts_valid",
        lambda candidate, **kwargs: True,
    )

    assert not run_v3._existing_results_match_context(
        [row],
        "L1-DIRECT",
        expected,
        config_path=run_v3.DEFAULT_CONFIG,
    )


def test_end_to_end_l3_smoke_is_isolated_and_never_loads_test(
    monkeypatch,
    tmp_path,
):
    original_load = datasets.load_adult
    loaded_splits: list[str] = []

    def tracked_load(split: str):
        loaded_splits.append(split)
        assert split == "train"
        return original_load(split)

    monkeypatch.setattr(run_v3.datasets, "load_adult", tracked_load)
    row = run_v3.execute_run(
        config_id="L3-IDENTITY",
        seed=0,
        smoke=True,
        artifacts_dir=tmp_path,
        verbose=False,
        command=["test-v3-smoke"],
    )

    assert loaded_splits == ["train"]
    assert row["run_kind"] == "smoke"
    assert row["status"] == "smoke_passed"
    assert row["activation"] == "none"
    assert row["depth"] == 3
    assert row["epochs"] == 2
    assert row["test_accuracy"] == ""
    assert row["flops_per_epoch"] == 1_544_654_481
    assert row["flops_total"] == 2 * 1_544_654_481
    assert row["inference_flops_total"] == 146_728_384
    assert not (tmp_path / "results.csv").exists()

    log_path = Path(row["log_path"])
    checkpoint_path = Path(row["checkpoint_path"])
    assert log_path.exists()
    assert checkpoint_path.exists()
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert events[0]["event"] == "run_attempt_started"
    started = next(event for event in events if event["event"] == "run_started")
    assert started["data"]["official_test_loaded"] is False
    assert started["training"]["evaluate_test"] is False
    assert started["model"]["identity_operation_created"] is False
    assert [event["epoch"] for event in events if event["event"] == "epoch"] == [
        1,
        2,
    ]
    diagnostics = [
        event for event in events if event["event"] == "diagnostic"
    ]
    assert [event["epoch"] for event in diagnostics] == [0, 1]
    assert all(event["equivalent_affine"] is True for event in diagnostics)
    assert events[-1]["event"] == "run_completed"

    with np.load(checkpoint_path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["metadata_json"].item()))
    assert manifest["depth"] == 3
    assert manifest["variable"] == "V3_linear_depth_without_activation"
