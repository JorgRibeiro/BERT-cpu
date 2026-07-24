"""Reproducible unit runner for Variable 3 of the Adult/q01 study.

Variable 3 changes the number of directly composed affine layers and never
creates an Identity activation. Scientific runs write only to
``experiments/v3`` and never load the official Adult test split. ``--smoke``
uses two epochs in isolated directories and never appends to ``results.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datasets  # noqa: E402
from bert_cpu import engine as cpu  # noqa: E402
from exercises import task_binary_classification as adult  # noqa: E402
from experiments import run_v1 as common  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/v3_linear_depth.json"
DEFAULT_ARTIFACTS = REPO_ROOT / "experiments/v3"
EXPECTED_BASE_COMMIT = "de000deae25b3939e2a392c07099ac0c2766679e"
REFERENCE_CONFIG_ID = "L1-DIRECT"
CONFIG_ORDER = ("L1-DIRECT", "L2-IDENTITY", "L3-IDENTITY")
DIAGNOSTIC_EPOCHS = (0, 1, 25, 50, 75, 100)

RESULT_FIELDS = (
    *common.RESULT_FIELDS[:5],
    "depth",
    "layer_sizes",
    *common.RESULT_FIELDS[5:],
)


def expected_run_id(
    config_id: str,
    seed: int,
    repetition: int = 1,
    smoke: bool = False,
) -> str:
    prefix = "SMOKE-" if smoke else ""
    return f"{prefix}{config_id}-s{seed}-r{repetition}"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "task",
        "variable",
        "phase",
        "base_commit",
        "architecture",
        "data",
        "training",
        "evaluation",
        "diagnostics",
        "artifacts",
        "configurations",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"configuration is missing fields: {sorted(missing)}")
    return config


def get_configuration(config: dict[str, Any], config_id: str) -> dict[str, Any]:
    matches = [item for item in config["configurations"] if item["id"] == config_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicated config id: {config_id!r}")
    return matches[0]


def validate_frozen_config(config: dict[str, Any]) -> None:
    """Reject drift from the V3 decisions registered before any smoke."""
    expected_identity = {
        "schema_version": 1,
        "status": "pre_experimental_confirmed",
        "created_at": "2026-07-24",
        "task": "adult_binary_classification",
        "variable": "V3_linear_depth_without_activation",
        "phase": "v3_train_validation",
        "branch": "q01-ativacoes-adult",
        "base_commit": EXPECTED_BASE_COMMIT,
    }
    if any(config.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("identity differs from the frozen V3 protocol")

    if config["architecture"] != {
        "input_features": 108,
        "hidden_features": 64,
        "output_classes": 2,
    }:
        raise ValueError("architecture constants differ from the V3 protocol")

    expected_data = {
        "raw_train_sha256": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
        "encoded_train_sha256": "b67c409c6f3fd0bfa90455e629af7d9b672260cb9267ae949103081a3fad1dfa",
        "split_sha256": "118bb0951fa1e0c4d88a4bbb493635d132492e2a52fb94698b792a99cec47bc0",
        "initial_weights_sha256": {
            "L1-DIRECT": {
                "0": "af18f27c17772dbcf215670e1d1239bb04fc98800e6f249486ac36346a0343e3",
                "1": "12aba40c05287347ab25056fbc7e90f8dee98952e52a5432dbd57f7637e02e6d",
                "2": "a517c716e0d004c39cc70eab934442bccb72256fc59df20bc29195e01ddb7e4f",
            },
            "L2-IDENTITY": {
                "0": "9b1fe0ce77aacdb4ec92847cb6026b61ec028f18a3948ee85873448d664f4596",
                "1": "d34dd1e6783a999faff6e235aefd23cd1e85a06bbe126423aec25bb1d8f867ae",
                "2": "0310d0d1c0c02d2fc0754b90f2d9b35150972361d64c9333bf52541163c26a29",
            },
            "L3-IDENTITY": {
                "0": "7fe853e4257c7962c0f1c0a90fb837612df76df7fb664235551b78163dcc51f4",
                "1": "98effc5a9c345922108a7f73eba681da943b913f505a98a8ce7c723b34b4e484",
                "2": "0053320cf6db0210c964ca9d9faa16c94249b4570e2bfdb7bf78e1ef939c6b00",
            },
        },
    }
    if config["data"] != expected_data:
        raise ValueError("data hashes differ from the frozen V3 protocol")

    expected_training = {
        "batching": "full_batch",
        "optimizer": "Adam",
        "learning_rate": 0.01,
        "epochs": 100,
        "validation_fraction": 0.2,
        "split_seed": 0,
        "model_seeds": [0, 1, 2],
    }
    if config["training"] != expected_training:
        raise ValueError("training differs from the frozen V3 protocol")

    expected_evaluation = {
        "primary_metric": "mean_validation_accuracy_at_epoch_100",
        "relevant_difference_percentage_points": 0.5,
        "required_seed_agreement": 2,
        "checkpoint": "epoch_100",
        "reference_configuration": "L1-DIRECT",
        "relu_comparison_configuration": "L2-IDENTITY",
        "relu_reference_id": "F-RELU",
        "decision_rules": {
            "H3a": (
                "contradicted_if_any_deeper_model_gains_at_least_delta_with_"
                "positive_sign_in_two_seeds_otherwise_not_contradicted"
            ),
            "H3b": (
                "sustained_only_if_parameters_and_flops_both_strictly_"
                "increase_L1_L2_L3"
            ),
            "H3c": (
                "sustained_if_return_strictly_decreases_L1_L2_L3_refuted_on_"
                "strict_reversal_otherwise_inconclusive"
            ),
            "H3d": (
                "sustained_if_relu_minus_L2_reaches_delta_and_is_positive_in_"
                "two_seeds_refuted_for_inverse_otherwise_inconclusive"
            ),
        },
        "test_policy": "final_phase_only_all_valid_configurations",
    }
    if config["evaluation"] != expected_evaluation:
        raise ValueError("evaluation differs from the frozen V3 protocol")

    expected_diagnostics = {
        "population": "fixed_validation_split",
        "epochs": list(DIAGNOSTIC_EPOCHS),
        "percentiles": [1, 5, 50, 95, 99],
        "affine_equivalence_rtol": 1e-12,
        "affine_equivalence_atol": 1e-12,
    }
    if config["diagnostics"] != expected_diagnostics:
        raise ValueError("diagnostics differ from the frozen V3 protocol")

    expected_artifacts = {
        "root": "experiments/v3",
        "log_format": "jsonl",
        "checkpoint_format": "npz_without_pickle",
        "smoke_appends_results": False,
        "scientific_run_requires_clean_source": True,
    }
    if config["artifacts"] != expected_artifacts:
        raise ValueError("artifact policy differs from the frozen V3 protocol")

    expected_configurations = [
        {
            "id": "L1-DIRECT",
            "depth": 1,
            "layer_sizes": [108, 2],
            "parameters": 218,
            "activation": "none",
            "activation_flops_per_element": 0,
            "expected_flops_per_epoch": 26_107_501,
            "expected_inference_flops_total": 2_839_232,
            "reference": True,
        },
        {
            "id": "L2-IDENTITY",
            "depth": 2,
            "layer_sizes": [108, 64, 2],
            "parameters": 7_106,
            "activation": "none",
            "activation_flops_per_element": 0,
            "expected_flops_per_epoch": 840_291_601,
            "expected_inference_flops_total": 92_548_544,
            "reference": False,
        },
        {
            "id": "L3-IDENTITY",
            "depth": 3,
            "layer_sizes": [108, 64, 64, 2],
            "parameters": 11_266,
            "activation": "none",
            "activation_flops_per_element": 0,
            "expected_flops_per_epoch": 1_544_654_481,
            "expected_inference_flops_total": 146_728_384,
            "reference": False,
        },
    ]
    if config["configurations"] != expected_configurations:
        raise ValueError("depth map differs from the frozen V3 protocol")


def _source_status() -> list[str]:
    watched = [
        "AGENTS.md",
        "Passo-a-passo.md",
        "PROJECT_STATUS.md",
        "requirements.txt",
        "bert_cpu",
        "datasets",
        "exercises/q01_activations.py",
        "exercises/task_binary_classification.py",
        "experiments/configs/v3_linear_depth.json",
        "experiments/hypotheses.md",
        "experiments/v3/protocol.md",
        "experiments/run_v1.py",
        "experiments/run_v3.py",
        "experiments/run_v3_all.py",
        "experiments/plot_v3.py",
        "test",
    ]
    output = common._git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *watched,
    )
    return output.splitlines() if output else []


def _source_files(config_path: Path) -> list[Path]:
    paths = list((REPO_ROOT / "bert_cpu").glob("*.py"))
    paths.extend(
        [
            REPO_ROOT / "datasets/__init__.py",
            REPO_ROOT / "datasets/loaders.py",
            REPO_ROOT / "exercises/q01_activations.py",
            REPO_ROOT / "exercises/task_binary_classification.py",
            REPO_ROOT / "experiments/run_v1.py",
            Path(__file__),
            REPO_ROOT / "experiments/run_v3_all.py",
            REPO_ROOT / "experiments/plot_v3.py",
            config_path,
            REPO_ROOT / "experiments/hypotheses.md",
            REPO_ROOT / "experiments/v3/protocol.md",
            REPO_ROOT / "AGENTS.md",
            REPO_ROOT / "Passo-a-passo.md",
            REPO_ROOT / "PROJECT_STATUS.md",
            REPO_ROOT / "requirements.txt",
        ]
    )
    paths.extend((REPO_ROOT / "test").glob("test_*v3*.py"))
    return [path for path in paths if path.exists()]


def _read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
            raise ValueError("V3 results.csv header does not match runner schema")
        return list(reader)


def load_results(path: Path | None = None) -> list[dict[str, str]]:
    return _read_results(path or (DEFAULT_ARTIFACTS / "results.csv"))


def _rewrite_results(path: Path, rows: list[dict[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=RESULT_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(
                {field: row.get(field, "") for field in RESULT_FIELDS}
                for row in rows
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        common._fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _append_result(path: Path, row: dict[str, Any]) -> None:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = _read_results(path)
        if any(existing["run_id"] == row["run_id"] for existing in rows):
            raise FileExistsError(f"run already recorded: {row['run_id']}")
        _rewrite_results(path, [*rows, row])


def _remove_result(path: Path, run_id: str) -> None:
    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = _read_results(path)
        remaining = [row for row in rows if row["run_id"] != run_id]
        if len(remaining) != len(rows):
            _rewrite_results(path, remaining)


def _architecture_manifest(
    config: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    return {
        **config["architecture"],
        "depth": selected["depth"],
        "layer_sizes": selected["layer_sizes"],
        "parameters": selected["parameters"],
    }


def affine_diagnostics(
    model: adult.AdultLinearClassifier,
    X: np.ndarray,
    *,
    epoch: int,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Verify and summarize the directly composed affine maps using NumPy."""
    if X.ndim != 2 or X.shape[0] != model.n_features:
        raise ValueError("diagnostic input does not match the model")

    current = X
    layers: list[dict[str, Any]] = []
    for index, layer in enumerate(model.layers, start=1):
        current = layer.weight.data[1:].T @ current + layer.weight.data[0][:, None]
        layers.append(
            {
                "index": index,
                "output_features": int(current.shape[0]),
                "output": common._summary(current),
            }
        )

    matrix, bias = adult.collapse_affine(model)
    collapsed = matrix @ X + bias[:, None]
    absolute_error = np.abs(current - collapsed)
    equivalent = bool(np.allclose(current, collapsed, rtol=rtol, atol=atol))
    return {
        "event": "diagnostic",
        "epoch": epoch,
        "population": "validation",
        "samples": int(X.shape[1]),
        "depth": model.depth,
        "layer_sizes": list(model.layer_sizes),
        "layers": layers,
        "collapsed_matrix": common._summary(matrix),
        "collapsed_bias": common._summary(bias),
        "collapsed_rank": int(np.linalg.matrix_rank(matrix)),
        "max_abs_error": float(absolute_error.max(initial=0.0)),
        "equivalent_affine": equivalent,
        "rtol": rtol,
        "atol": atol,
    }


def _load_v3_checkpoint(
    path: Path,
    model: adult.AdultLinearClassifier,
    *,
    expected_sha256: str,
    expected_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate a V3 checkpoint completely before restoring any parameter."""
    if common.sha256_file(path) != expected_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    names_and_parameters = list(model.named_parameters())
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["metadata_json"].item()))
        expected_names = [name for name, _ in names_and_parameters]
        expected_shapes = {
            name: list(parameter.shape) for name, parameter in names_and_parameters
        }
        if manifest.get("parameter_names") != expected_names:
            raise ValueError("checkpoint parameter names do not match model")
        if manifest.get("parameter_shapes") != expected_shapes:
            raise ValueError("checkpoint parameter shapes do not match model")
        if any(manifest.get(key) != value for key, value in expected_manifest.items()):
            raise ValueError("checkpoint manifest differs from expected V3 metadata")

        stored_parameters: list[np.ndarray] = []
        for index, (_, parameter) in enumerate(names_and_parameters):
            stored = archive[f"parameter_{index}"].copy()
            if stored.shape != parameter.shape or stored.dtype != parameter.data.dtype:
                raise ValueError("checkpoint parameter shape or dtype mismatch")
            if not np.isfinite(stored).all():
                raise ValueError("checkpoint contains NaN or Inf")
            stored_parameters.append(stored)

    if common.hash_arrays(*stored_parameters) != manifest.get("final_weights_hash"):
        raise ValueError("checkpoint final-weights hash mismatch")
    for stored, (_, parameter) in zip(stored_parameters, names_and_parameters):
        parameter.data[...] = stored
    return manifest


def result_artifacts_valid(
    row: dict[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> bool:
    """Validate one completed V3 row and all immutable evidence it references."""
    try:
        config_path = config_path.resolve()
        config = load_config(config_path)
        validate_frozen_config(config)
        selected = get_configuration(config, str(row["config_id"]))
        seed = int(row["seed"])
        repetition = int(row["repetition"])
        epochs = int(config["training"]["epochs"])
        expected_epoch_flops = int(selected["expected_flops_per_epoch"])
        expected_total_flops = expected_epoch_flops * epochs
        expected_inference_flops = int(selected["expected_inference_flops_total"])
        recorded_sizes = json.loads(str(row["layer_sizes"]))

        if (
            row["task"] != config["task"]
            or row["variable"] != config["variable"]
            or row["phase"] != config["phase"]
            or row["run_kind"] != "scientific"
            or row["purpose"] != "primary"
            or row["status"] != "completed_valid"
            or int(row["epochs"]) != epochs
            or str(row.get("test_accuracy", "")).strip()
            or row["activation"] != "none"
            or int(row["depth"]) != int(selected["depth"])
            or recorded_sizes != selected["layer_sizes"]
            or seed not in config["training"]["model_seeds"]
            or repetition != 1
            or row["run_id"] != expected_run_id(str(row["config_id"]), seed)
            or row["config_hash"] != common.sha256_file(config_path)
            or int(row["split_seed"]) != int(config["training"]["split_seed"])
            or row["split_hash"] != config["data"]["split_sha256"]
            or row["dataset_hash"] != config["data"]["encoded_train_sha256"]
            or row["initial_weights_hash"]
            != config["data"]["initial_weights_sha256"][str(row["config_id"])][
                str(seed)
            ]
            or int(row["parameters"]) != int(selected["parameters"])
            or int(row["train_samples"]) != 26_049
            or int(row["val_samples"]) != 6_512
            or int(row["flops_per_epoch"]) != expected_epoch_flops
            or int(row["flops_total"]) != expected_total_flops
            or int(row["inference_flops_total"]) != expected_inference_flops
            or not np.isclose(
                float(row["gflops_total"]),
                expected_total_flops / 1e9,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                float(row["inference_flops_per_sample"]),
                expected_inference_flops / 6_512,
                rtol=0.0,
                atol=1e-12,
            )
            or "official_test_not_loaded" not in str(row["notes"])
        ):
            return False
        if not common._result_artifacts_valid(row):
            return False

        checkpoint_path = common._recorded_path(str(row["checkpoint_path"]))
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive["metadata_json"].item()))
        if (
            manifest.get("variable") != row["variable"]
            or manifest.get("phase") != row["phase"]
            or manifest.get("purpose") != row["purpose"]
            or manifest.get("depth") != selected["depth"]
            or manifest.get("layer_sizes") != selected["layer_sizes"]
            or manifest.get("epoch") != epochs
            or manifest.get("architecture")
            != _architecture_manifest(config, selected)
            or manifest.get("training") != config["training"]
            or manifest.get("configuration") != selected
        ):
            return False

        log_path = common._recorded_path(str(row["log_path"]))
        events = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        started = [event for event in events if event.get("event") == "run_started"]
        epoch_events = [event for event in events if event.get("event") == "epoch"]
        diagnostics = [
            event for event in events if event.get("event") == "diagnostic"
        ]
        if len(started) != 1:
            return False
        start = started[0]
        if (
            start.get("configuration") != selected
            or start.get("data", {}).get("split_loaded") != "train"
            or start.get("data", {}).get("official_test_loaded") is not False
            or start.get("training", {}).get("evaluate_test") is not False
            or start.get("training", {}).get("epochs") != epochs
        ):
            return False

        metric_fields = (
            "train_loss",
            "val_loss",
            "train_accuracy",
            "val_accuracy",
        )
        metric_values = np.asarray(
            [
                float(event[field])
                for event in epoch_events
                for field in metric_fields
            ]
        )
        final = epoch_events[-1]
        return (
            [event["epoch"] for event in epoch_events]
            == list(range(1, epochs + 1))
            and all(
                int(event["flops"]) == expected_epoch_flops
                for event in epoch_events
            )
            and np.isfinite(metric_values).all()
            and float(row["train_loss_final"]) == float(final["train_loss"])
            and float(row["val_loss_final"]) == float(final["val_loss"])
            and float(row["train_accuracy"]) == float(final["train_accuracy"])
            and float(row["val_accuracy"]) == float(final["val_accuracy"])
            and [event["epoch"] for event in diagnostics]
            == list(DIAGNOSTIC_EPOCHS)
            and all(
                event.get("population") == "validation"
                and event.get("equivalent_affine") is True
                and int(event["depth"]) == int(selected["depth"])
                and int(event["samples"]) == 6_512
                and np.isfinite(float(event["max_abs_error"]))
                for event in diagnostics
            )
        )
    except (
        OSError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        OverflowError,
        json.JSONDecodeError,
    ):
        return False


def _configuration_complete(
    rows: Sequence[dict[str, str]],
    config_id: str,
    *,
    verify_artifacts: bool,
    config_path: Path = DEFAULT_CONFIG,
) -> bool:
    selected = [
        row
        for row in rows
        if row.get("config_id") == config_id
        and row.get("status") == "completed_valid"
        and (
            not verify_artifacts
            or result_artifacts_valid(row, config_path=config_path)
        )
    ]
    return (
        len(selected) == 3
        and {int(row["seed"]) for row in selected} == {0, 1, 2}
        and all(int(row["repetition"]) == 1 for row in selected)
    )


def _required_predecessors(config_id: str) -> tuple[str, ...]:
    index = CONFIG_ORDER.index(config_id)
    return CONFIG_ORDER[:index]


def _existing_results_match_context(
    rows: Sequence[dict[str, str]],
    config_id: str,
    expected: dict[str, Any],
    *,
    config_path: Path,
) -> bool:
    required = _required_predecessors(config_id)
    run_ids = [row.get("run_id", "") for row in rows]
    return (
        len(run_ids) == len(set(run_ids))
        and all(
            result_artifacts_valid(row, config_path=config_path)
            for row in rows
        )
        and all(
            _configuration_complete(
                rows,
                predecessor,
                verify_artifacts=True,
                config_path=config_path,
            )
            for predecessor in required
        )
        and all(
            all(row[field] == str(value) for field, value in expected.items())
            for row in rows
        )
    )


def execute_run(
    *,
    config_id: str,
    seed: int,
    repetition: int = 1,
    smoke: bool = False,
    config_path: Path = DEFAULT_CONFIG,
    artifacts_dir: Path = DEFAULT_ARTIFACTS,
    verbose: bool = True,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Execute one isolated V3 configuration and persist its evidence."""
    config_path = config_path.resolve()
    artifacts_dir = artifacts_dir.resolve()
    config = load_config(config_path)
    validate_frozen_config(config)
    selected = get_configuration(config, config_id)
    training_config = config["training"]

    if seed not in training_config["model_seeds"]:
        raise ValueError(f"seed {seed} is not in the frozen configuration")
    if repetition != 1:
        raise ValueError("V3 permits exactly one repetition per seed")

    epochs = 2 if smoke else int(training_config["epochs"])
    run_kind = "smoke" if smoke else "scientific"
    purpose = "smoke_validation" if smoke else "primary"
    run_id = expected_run_id(config_id, seed, repetition, smoke)
    subdirectory = "smoke" if smoke else ""
    log_path = artifacts_dir / "logs" / subdirectory / f"{run_id}.jsonl"
    checkpoint_path = (
        artifacts_dir / "checkpoints" / subdirectory / f"{run_id}.npz"
    )
    results_path = artifacts_dir / "results.csv"

    if log_path.exists() or checkpoint_path.exists():
        raise FileExistsError(f"artifacts already exist for {run_id}")
    existing_results = load_results(results_path)
    if any(row["run_id"] == run_id for row in existing_results):
        raise FileExistsError(f"run already exists in results.csv: {run_id}")
    missing = [
        predecessor
        for predecessor in _required_predecessors(config_id)
        if not _configuration_complete(
            existing_results,
            predecessor,
            verify_artifacts=True,
            config_path=config_path,
        )
    ]
    if not smoke and missing:
        raise RuntimeError(
            f"V3 run is blocked until these configurations are complete: {missing}"
        )

    branch = common._git("branch", "--show-current")
    commit = common._git("rev-parse", "HEAD")
    source_status = _source_status()
    if not smoke and branch != config["branch"]:
        raise RuntimeError(f"scientific run requires branch {config['branch']!r}")
    if not smoke and source_status:
        raise RuntimeError(f"scientific run requires clean source state: {source_status}")

    code_state_hash = common._hash_files(_source_files(config_path))
    config_hash = common.sha256_file(config_path)
    environment = common._environment_metadata()
    environment_hash = common._hash_json(environment)
    reservation_path = common._reserve_run(artifacts_dir, run_id)

    start_event: dict[str, Any] | None = {
        "event": "run_attempt_started",
        "schema_version": 1,
        "timestamp_utc": common._utc_now(),
        "run_id": run_id,
        "run_kind": run_kind,
        "phase": config["phase"],
        "purpose": purpose,
        "status": "initializing",
        "config_id": config_id,
        "activation": "none",
        "depth": selected["depth"],
        "layer_sizes": selected["layer_sizes"],
        "seed": seed,
        "repetition": repetition,
        "command": list(command or sys.argv),
        "branch": branch,
        "commit": commit,
        "code_state_hash": code_state_hash,
        "config_hash": config_hash,
        "environment_hash": environment_hash,
        "source_status": source_status,
        "reservation_path": common._display_path(reservation_path),
    }
    common._append_jsonl(log_path, start_event)

    try:
        train_dataset = datasets.load_adult("train")
        split_seed = int(training_config["split_seed"])
        train_indices, val_indices = adult.train_val_indices(
            train_dataset.n_samples,
            val_frac=float(training_config["validation_fraction"]),
            split_seed=split_seed,
        )
        X_train = train_dataset.X[:, train_indices]
        y_train = train_dataset.y[train_indices]
        X_val = train_dataset.X[:, val_indices]
        y_val = train_dataset.y[val_indices]

        if train_dataset.n_features != int(config["architecture"]["input_features"]):
            raise RuntimeError("Adult feature count differs from frozen configuration")
        model = adult.build_linear_model(
            train_dataset.n_features,
            depth=int(selected["depth"]),
            model_seed=seed,
            hidden=int(config["architecture"]["hidden_features"]),
        )
        parameters = adult.parameter_count(model)
        if list(model.layer_sizes) != selected["layer_sizes"]:
            raise RuntimeError("model layer sizes differ from frozen configuration")
        if parameters != int(selected["parameters"]):
            raise RuntimeError("model parameter count differs from frozen configuration")

        dataset_hash = common.hash_arrays(train_dataset.X, train_dataset.y)
        split_hash = common.hash_arrays(train_indices, val_indices)
        initial_weights_hash = common.parameter_hash(model)
        raw_train_hash = common.sha256_file(
            REPO_ROOT / "datasets/adult/adult.data"
        )
        frozen_data = config["data"]
        if raw_train_hash != frozen_data["raw_train_sha256"]:
            raise RuntimeError("raw Adult train hash differs from frozen configuration")
        if dataset_hash != frozen_data["encoded_train_sha256"]:
            raise RuntimeError("encoded Adult train hash differs from frozen configuration")
        if split_hash != frozen_data["split_sha256"]:
            raise RuntimeError("split hash differs from frozen configuration")
        if (
            initial_weights_hash
            != frozen_data["initial_weights_sha256"][config_id][str(seed)]
        ):
            raise RuntimeError("initial weights hash differs from frozen configuration")

        majority_accuracy = float(np.bincount(y_val).max() / len(y_val))
        context = {
            "task": config["task"],
            "variable": config["variable"],
            "run_kind": "scientific",
            "phase": config["phase"],
            "purpose": "primary",
            "status": "completed_valid",
            "branch": branch,
            "commit": commit,
            "base_commit": config["base_commit"],
            "code_state_hash": code_state_hash,
            "config_hash": config_hash,
            "environment_hash": environment_hash,
            "dataset_hash": dataset_hash,
            "split_seed": split_seed,
            "split_hash": split_hash,
            "epochs": epochs,
            "optimizer": training_config["optimizer"],
            "learning_rate": training_config["learning_rate"],
            "train_samples": len(y_train),
            "val_samples": len(y_val),
        }
        if not smoke and not _existing_results_match_context(
            existing_results,
            config_id,
            context,
            config_path=config_path,
        ):
            raise RuntimeError(
                "existing V3 runs do not match the current context or artifacts"
            )

        start_event = {
            "event": "run_started",
            "schema_version": 1,
            "timestamp_utc": common._utc_now(),
            "run_id": run_id,
            "run_kind": run_kind,
            "phase": config["phase"],
            "purpose": purpose,
            "status": "running",
            "task": config["task"],
            "variable": config["variable"],
            "config_id": config_id,
            "activation": "none",
            "depth": selected["depth"],
            "layer_sizes": selected["layer_sizes"],
            "configuration": selected,
            "seed": seed,
            "repetition": repetition,
            "command": list(command or sys.argv),
            "provenance": {
                "branch": branch,
                "commit": commit,
                "base_commit": config["base_commit"],
                "source_dirty": bool(source_status),
                "source_status": source_status,
                "code_state_hash": code_state_hash,
                "config_path": common._display_path(config_path),
                "config_hash": config_hash,
                "environment_hash": environment_hash,
            },
            "environment": environment,
            "data": {
                "split_loaded": "train",
                "official_test_loaded": False,
                "raw_train_hash": raw_train_hash,
                "encoded_train_hash": dataset_hash,
                "features": train_dataset.n_features,
                "samples": train_dataset.n_samples,
                "train_samples": len(train_indices),
                "validation_samples": len(val_indices),
                "split_seed": split_seed,
                "split_hash": split_hash,
                "validation_majority_accuracy": majority_accuracy,
                "preprocessing_limitation": (
                    "encoder_fit_on_official_train_before_holdout"
                ),
            },
            "model": {
                "architecture": _architecture_manifest(config, selected),
                "parameters": parameters,
                "initial_weights_hash": initial_weights_hash,
                "identity_operation_created": False,
            },
            "training": {
                "epochs": epochs,
                "optimizer": training_config["optimizer"],
                "learning_rate": training_config["learning_rate"],
                "batching": training_config["batching"],
                "train_loss_timing": "before_adam_step",
                "validation_loss_timing": "after_adam_step",
                "accuracy_timing": "after_adam_step_outside_flop_window",
                "evaluate_test": False,
            },
            "diagnostics": config["diagnostics"],
        }
        common._append_jsonl(log_path, start_event)

        diagnostic_options = {
            "rtol": float(config["diagnostics"]["affine_equivalence_rtol"]),
            "atol": float(config["diagnostics"]["affine_equivalence_atol"]),
        }
        initial_flops = cpu.flop_count()
        initial_parameter_hash = common.parameter_hash(model)
        initial_rng = np.random.get_state()
        diagnostic = affine_diagnostics(
            model,
            X_val,
            epoch=0,
            **diagnostic_options,
        )
        if not diagnostic["equivalent_affine"]:
            raise RuntimeError("epoch-0 affine collapse check failed")
        if cpu.flop_count() != initial_flops:
            raise RuntimeError("epoch-0 diagnostics changed the FLOP counter")
        if common.parameter_hash(model) != initial_parameter_hash:
            raise RuntimeError("epoch-0 diagnostics changed model parameters")
        if not common._rng_state_equal(initial_rng, np.random.get_state()):
            raise RuntimeError("epoch-0 diagnostics changed RNG state")
        common._append_jsonl(log_path, diagnostic)
        cpu.reset_flops()

        def record_epoch(
            metrics: adult.EpochMetrics,
            current_model: adult.AdultLinearClassifier,
        ) -> None:
            common._append_jsonl(log_path, {"event": "epoch", **asdict(metrics)})
            if metrics.epoch not in DIAGNOSTIC_EPOCHS:
                return
            flops_before = cpu.flop_count()
            weights_before = common.parameter_hash(current_model)
            rng_before = np.random.get_state()
            current_diagnostic = affine_diagnostics(
                current_model,
                X_val,
                epoch=metrics.epoch,
                **diagnostic_options,
            )
            if not current_diagnostic["equivalent_affine"]:
                raise RuntimeError("affine collapse check failed")
            if cpu.flop_count() != flops_before:
                raise RuntimeError("diagnostics changed the FLOP counter")
            if common.parameter_hash(current_model) != weights_before:
                raise RuntimeError("diagnostics changed model parameters")
            if not common._rng_state_equal(rng_before, np.random.get_state()):
                raise RuntimeError("diagnostics changed RNG state")
            common._append_jsonl(log_path, current_diagnostic)
            cpu.reset_flops()

        training = adult.train(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            lr=float(training_config["learning_rate"]),
            verbose=verbose,
            epoch_callback=record_epoch,
        )
        flops_per_epoch = common._validate_history(
            training,
            epochs,
            int(selected["expected_flops_per_epoch"]),
        )
        if not all(
            np.isfinite(parameter.data).all() for parameter in model.parameters()
        ):
            raise FloatingPointError("final weights contain NaN or Inf")

        final_weights_hash = common.parameter_hash(model)
        cpu.reset_flops()
        model(cpu.Tensor(X_val, requires_grad=False))
        inference_flops_total = cpu.flop_count()
        if inference_flops_total != int(selected["expected_inference_flops_total"]):
            raise RuntimeError("inference FLOPs differ from frozen value")
        inference_flops_per_sample = inference_flops_total / len(y_val)

        checkpoint_manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "run_kind": run_kind,
            "purpose": purpose,
            "variable": config["variable"],
            "phase": config["phase"],
            "epoch": epochs,
            "config_id": config_id,
            "activation": "none",
            "depth": selected["depth"],
            "layer_sizes": selected["layer_sizes"],
            "seed": seed,
            "repetition": repetition,
            "split_seed": split_seed,
            "split_hash": split_hash,
            "config_hash": config_hash,
            "environment_hash": environment_hash,
            "commit": commit,
            "code_state_hash": code_state_hash,
            "initial_weights_hash": initial_weights_hash,
            "final_weights_hash": final_weights_hash,
            "architecture": _architecture_manifest(config, selected),
            "training": training_config,
            "configuration": selected,
        }
        checkpoint_hash = common.save_checkpoint(
            checkpoint_path,
            model,
            checkpoint_manifest,
        )
        verification_model = adult.build_linear_model(
            train_dataset.n_features,
            depth=int(selected["depth"]),
            model_seed=seed,
            hidden=int(config["architecture"]["hidden_features"]),
        )
        loaded_manifest = _load_v3_checkpoint(
            checkpoint_path,
            verification_model,
            expected_sha256=checkpoint_hash,
            expected_manifest=checkpoint_manifest,
        )
        expected_loaded_manifest = {
            **checkpoint_manifest,
            "parameter_names": [name for name, _ in model.named_parameters()],
            "parameter_shapes": {
                name: list(parameter.shape)
                for name, parameter in model.named_parameters()
            },
        }
        if loaded_manifest != expected_loaded_manifest:
            raise RuntimeError("checkpoint manifest round-trip failed")
        if common.parameter_hash(verification_model) != final_weights_hash:
            raise RuntimeError("checkpoint weights round-trip failed")

        final_metrics = training.history[-1]
        terminal_status = "smoke_passed" if smoke else "completed_valid"
        row = {
            "run_id": run_id,
            "task": config["task"],
            "variable": config["variable"],
            "config_id": config_id,
            "activation": "none",
            "depth": selected["depth"],
            "layer_sizes": json.dumps(
                selected["layer_sizes"],
                separators=(",", ":"),
            ),
            "seed": seed,
            "repetition": repetition,
            "run_kind": run_kind,
            "phase": config["phase"],
            "purpose": purpose,
            "status": terminal_status,
            "branch": branch,
            "commit": commit,
            "base_commit": config["base_commit"],
            "code_state_hash": code_state_hash,
            "config_hash": config_hash,
            "environment_hash": environment_hash,
            "dataset_hash": dataset_hash,
            "split_seed": split_seed,
            "split_hash": split_hash,
            "initial_weights_hash": initial_weights_hash,
            "final_weights_hash": final_weights_hash,
            "epochs": epochs,
            "optimizer": training_config["optimizer"],
            "learning_rate": training_config["learning_rate"],
            "train_samples": len(y_train),
            "val_samples": len(y_val),
            "parameters": parameters,
            "train_loss_final": final_metrics.train_loss,
            "val_loss_final": final_metrics.val_loss,
            "train_accuracy": final_metrics.train_accuracy,
            "val_accuracy": final_metrics.val_accuracy,
            "test_accuracy": "",
            "flops_per_epoch": flops_per_epoch,
            "flops_total": training.total_flops,
            "gflops_total": training.total_flops / 1e9,
            "inference_flops_total": inference_flops_total,
            "inference_flops_per_sample": inference_flops_per_sample,
            "checkpoint_path": common._display_path(checkpoint_path),
            "checkpoint_hash": checkpoint_hash,
            "log_path": common._display_path(log_path),
            "notes": (
                "official_test_not_loaded; smoke_not_scientific"
                if smoke
                else "official_test_not_loaded"
            ),
        }
        common._append_jsonl(
            log_path,
            {
                "event": "artifacts_verified",
                "timestamp_utc": common._utc_now(),
                "status": "artifacts_verified",
                "summary": row,
                "checkpoint_verified": True,
            },
        )
        if not smoke:
            _append_result(results_path, row)
        try:
            common._append_jsonl(
                log_path,
                {
                    "event": "run_completed",
                    "timestamp_utc": common._utc_now(),
                    "status": terminal_status,
                    "result_registered": not smoke,
                },
            )
        except Exception:
            if not smoke:
                _remove_result(results_path, run_id)
            raise
        return row
    except Exception as error:
        if start_event is not None:
            common._append_jsonl(
                log_path,
                {
                    "event": "run_failed",
                    "timestamp_utc": common._utc_now(),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one frozen Variable 3 affine-depth configuration.",
    )
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repetition", type=int, default=1)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run two non-scientific epochs and do not modify results.csv",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    command = [sys.executable, "-m", "experiments.run_v3", *(argv or sys.argv[1:])]
    result = execute_run(
        config_id=args.config_id,
        seed=args.seed,
        repetition=args.repetition,
        smoke=args.smoke,
        config_path=args.config,
        artifacts_dir=args.artifacts_dir,
        verbose=not args.quiet,
        command=command,
    )
    print(
        f"{result['run_id']}: {result['status']} | "
        f"depth={int(result['depth'])} | "
        f"val={result['val_accuracy']:.4f} | FLOPs={result['flops_total']:,}"
    )
    return result


if __name__ == "__main__":
    main()
