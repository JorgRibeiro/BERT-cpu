"""Tests for the all-checkpoint official Adult evaluator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import datasets
from experiments import evaluate_official_test as official


def test_official_grid_is_exact_and_excludes_determinism_repeat():
    specs = official.collect_checkpoint_specs()

    assert len(specs) == 33
    assert tuple(spec.run_id for spec in specs) == official.EXPECTED_RUN_IDS
    assert "F-RELU-s0-r2" not in {spec.run_id for spec in specs}
    assert {spec.seed for spec in specs} == {0, 1, 2}


def test_preflight_uses_train_but_never_official_test(monkeypatch):
    original = datasets.load_adult
    observed: list[str] = []

    def tracked_load(split):
        observed.append(split)
        if split == "test":
            raise AssertionError("preflight accessed official test")
        return original(split)

    monkeypatch.setattr(official.datasets, "load_adult", tracked_load)
    manifest, specs = official.build_input_manifest()

    assert observed == ["train"]
    assert manifest["status"] == "preflight_valid"
    assert manifest["checkpoint_count"] == 33
    assert len(specs) == 33


def test_preflight_reproduces_all_validation_metrics_and_flops():
    manifest, _ = official.build_input_manifest()
    specs_by_id = {
        spec.run_id: spec
        for spec in official.collect_checkpoint_specs()
    }

    for item in manifest["checkpoints"]:
        assert (
            item["validation_reproduced_accuracy"]
            == item["source_val_accuracy"]
        )
        expected = specs_by_id[item["source_run_id"]]
        assert item["validation_reproduced_flops"] == int(
            expected.row["inference_flops_total"]
        )


def test_result_validation_rejects_incomplete_grid():
    with pytest.raises(ValueError, match="exactly 33"):
        official._validate_result_rows([], "OFFICIAL-fixture")


def test_parser_exposes_no_configuration_or_seed_subset():
    parser = official._build_parser()
    destinations = {action.dest for action in parser._actions}

    assert "config_id" not in destinations
    assert "seed" not in destinations
    assert "preflight" in destinations
    assert "evaluate_official_test" in destinations


def test_test_metadata_rejects_wrong_shape_without_reading_raw_file():
    class FakeDataset:
        split = "test"
        X = np.zeros((108, 2), dtype=np.float64)
        y = np.zeros(2, dtype=np.int64)

    with pytest.raises(ValueError, match="unexpected schema"):
        official._official_test_metadata(FakeDataset())


def test_prepare_input_manifest_is_idempotent(tmp_path, monkeypatch):
    manifest = {
        "evaluation_id": "OFFICIAL-fixture",
        "checkpoint_count": 33,
        "status": "preflight_valid",
    }
    monkeypatch.setattr(
        official,
        "build_input_manifest",
        lambda: (manifest, tuple()),
    )

    first = official.prepare_input_manifest(tmp_path)
    second = official.prepare_input_manifest(tmp_path)

    assert first == second == manifest
    assert (tmp_path / official.INPUT_MANIFEST_NAME).is_file()


def test_prepare_input_manifest_refuses_partial_artifacts(tmp_path, monkeypatch):
    manifest = {
        "evaluation_id": "OFFICIAL-fixture",
        "checkpoint_count": 33,
        "status": "preflight_valid",
    }
    monkeypatch.setattr(
        official,
        "build_input_manifest",
        lambda: (manifest, tuple()),
    )
    (tmp_path / official.LOG_NAME).write_text("partial\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="partial"):
        official.prepare_input_manifest(tmp_path)


def _fake_test_metadata(feature_schema_hash):
    return {
        "raw_path": "datasets/adult/adult.test",
        "raw_sha256": official.EXPECTED_RAW_TEST_SHA256,
        "git_blob": official.EXPECTED_RAW_TEST_GIT_BLOB,
        "encoded_hash": "encoded-test-fixture",
        "features_hash": "features-fixture",
        "labels_hash": "labels-fixture",
        "feature_schema_hash": feature_schema_hash,
        "features": 108,
        "samples": official.EXPECTED_TEST_SAMPLES,
        "positive_fraction": 0.25,
        "majority_accuracy": 0.75,
    }


class _FakeTestDataset:
    split = "test"
    X = object()
    y = object()
    n_features = 108
    n_samples = official.EXPECTED_TEST_SAMPLES


def _fake_forward(spec, X, y):
    per_sample = int(
        int(spec.row["inference_flops_total"])
        / official.EXPECTED_VALIDATION_SAMPLES
    )
    correct = 13_000 + spec.seed
    prediction_identity = (
        f"softplus-pair-seed-{spec.seed}"
        if spec.config_id in {"F-SOFTPLUS", "S-BETA-1"}
        else spec.run_id
    )
    return {
        "correct": correct,
        "accuracy": correct / official.EXPECTED_TEST_SAMPLES,
        "flops": per_sample * official.EXPECTED_TEST_SAMPLES,
        "predictions_hash": official.run_v1._hash_json(prediction_identity),
    }


def test_complete_path_loads_test_once_and_rerun_loads_zero(
    tmp_path,
    monkeypatch,
):
    manifest, specs = official.build_input_manifest()
    monkeypatch.setattr(
        official,
        "build_input_manifest",
        lambda: (manifest, specs),
    )
    official.prepare_input_manifest(tmp_path)
    monkeypatch.setattr(official, "_forward_once", _fake_forward)
    monkeypatch.setattr(
        official,
        "_official_test_metadata",
        lambda dataset: _fake_test_metadata(
            manifest["training_data"]["feature_schema_hash"]
        ),
    )
    loads: list[str] = []

    def load_once(split):
        loads.append(split)
        return _FakeTestDataset()

    output, rows = official.run_official_evaluation(
        tmp_path,
        test_loader=load_once,
    )

    assert loads == ["test"]
    assert output["official_test_loaded_count"] == 1
    assert len(rows) == 33

    def must_not_load(split):
        raise AssertionError("idempotent rerun loaded official test")

    repeated_output, repeated_rows = official.run_official_evaluation(
        tmp_path,
        test_loader=must_not_load,
    )
    assert repeated_output == output
    assert repeated_rows == rows


def test_failure_after_test_load_is_closed_and_never_reloads(
    tmp_path,
    monkeypatch,
):
    manifest, specs = official.build_input_manifest()
    monkeypatch.setattr(
        official,
        "build_input_manifest",
        lambda: (manifest, specs),
    )
    official.prepare_input_manifest(tmp_path)
    monkeypatch.setattr(
        official,
        "_official_test_metadata",
        lambda dataset: _fake_test_metadata(
            manifest["training_data"]["feature_schema_hash"]
        ),
    )
    monkeypatch.setattr(
        official,
        "_forward_once",
        lambda spec, X, y: (_ for _ in ()).throw(
            RuntimeError("fixture forward failure")
        ),
    )
    loads: list[str] = []

    def tracked_loader(split):
        loads.append(split)
        return _FakeTestDataset()

    with pytest.raises(RuntimeError, match="fixture forward failure"):
        official.run_official_evaluation(
            tmp_path,
            test_loader=tracked_loader,
        )
    assert loads == ["test"]
    assert (tmp_path / official.LOG_NAME).is_file()
    assert not (tmp_path / official.OUTPUT_MANIFEST_NAME).exists()

    with pytest.raises(RuntimeError, match="partial"):
        official.run_official_evaluation(
            tmp_path,
            test_loader=tracked_loader,
        )
    assert loads == ["test"]


def test_source_result_files_remain_outside_final_artifacts():
    paths = official._artifact_paths(Path("experiments/final_evaluation"))

    assert paths["results"].name == "results.csv"
    assert paths["results"] != Path("experiments/results.csv")
    assert paths["results"] != Path("experiments/v2/results.csv")
    assert paths["results"] != Path("experiments/v3/results.csv")
