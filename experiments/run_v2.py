"""Reproducible unit runner for Variable 2 of the Adult/q01 study.

Variable 2 changes only the fixed curvature of Softplus. Scientific runs use a
frozen configuration, write to ``experiments/v2`` and never load the official
Adult test split. ``--smoke`` runs two epochs in an isolated subdirectory and
does not append to ``results.csv``.
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


DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/v2_softplus_curvature.json"
DEFAULT_ARTIFACTS = REPO_ROOT / "experiments/v2"
EXPECTED_BASE_COMMIT = "07243dc6591f5c30c654813313b43fb6159e7fb0"
REFERENCE_CONFIG_ID = "S-BETA-1"
DIAGNOSTIC_EPOCHS = (0, 1, 25, 50, 75, 100)
LOW_DERIVATIVE_THRESHOLD = 0.05
HIGH_DERIVATIVE_THRESHOLD = 0.95

RESULT_FIELDS = (
    *common.RESULT_FIELDS[:5],
    "beta",
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
    """Reject any drift from the V2 decisions registered before the runs."""
    expected_identity = {
        "schema_version": 1,
        "status": "pre_experimental_confirmed",
        "created_at": "2026-07-23",
        "task": "adult_binary_classification",
        "variable": "V2_softplus_curvature",
        "phase": "v2_train_validation",
        "branch": "q01-ativacoes-adult",
        "base_commit": EXPECTED_BASE_COMMIT,
    }
    if any(config.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("identity differs from the frozen V2 protocol")

    expected_architecture = {
        "input_features": 108,
        "hidden_features": 64,
        "output_classes": 2,
        "parameters": 7106,
    }
    if config["architecture"] != expected_architecture:
        raise ValueError("architecture differs from the frozen V2 protocol")

    expected_data = {
        "raw_train_sha256": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
        "encoded_train_sha256": "b67c409c6f3fd0bfa90455e629af7d9b672260cb9267ae949103081a3fad1dfa",
        "split_sha256": "118bb0951fa1e0c4d88a4bbb493635d132492e2a52fb94698b792a99cec47bc0",
        "initial_weights_sha256": {
            "0": "9b1fe0ce77aacdb4ec92847cb6026b61ec028f18a3948ee85873448d664f4596",
            "1": "d34dd1e6783a999faff6e235aefd23cd1e85a06bbe126423aec25bb1d8f867ae",
            "2": "0310d0d1c0c02d2fc0754b90f2d9b35150972361d64c9333bf52541163c26a29",
        },
    }
    if config["data"] != expected_data:
        raise ValueError("data hashes differ from the frozen V2 protocol")

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
        raise ValueError("training differs from the frozen V2 protocol")

    expected_evaluation = {
        "primary_metric": "mean_validation_accuracy_at_epoch_100",
        "relevant_difference_percentage_points": 0.5,
        "required_seed_agreement": 2,
        "central_group": ["S-BETA-1", "S-BETA-2"],
        "extreme_group": ["S-BETA-0.5", "S-BETA-5"],
        "tie_policy": "all_co_winner_comparisons_must_agree_otherwise_inconclusive",
        "checkpoint": "epoch_100",
        "test_policy": "final_phase_only_all_valid_configurations",
    }
    if config["evaluation"] != expected_evaluation:
        raise ValueError("evaluation differs from the frozen V2 protocol")

    expected_diagnostics = {
        "population": "fixed_validation_split",
        "epochs": list(DIAGNOSTIC_EPOCHS),
        "percentiles": [1, 5, 50, 95, 99],
        "near_zero_threshold": 1e-6,
        "low_derivative_threshold": LOW_DERIVATIVE_THRESHOLD,
        "high_derivative_threshold": HIGH_DERIVATIVE_THRESHOLD,
    }
    if config["diagnostics"] != expected_diagnostics:
        raise ValueError("diagnostics differ from the frozen V2 protocol")

    expected_artifacts = {
        "root": "experiments/v2",
        "log_format": "jsonl",
        "checkpoint_format": "npz_without_pickle",
        "smoke_appends_results": False,
        "scientific_run_requires_clean_source": True,
    }
    if config["artifacts"] != expected_artifacts:
        raise ValueError("artifact policy differs from the frozen V2 protocol")

    expected_configurations = [
        {
            "id": "S-BETA-0.5",
            "activation": "softplus_beta",
            "beta": 0.5,
            "activation_flops_per_element": 5,
            "expected_flops_per_epoch": 850711121,
            "expected_inference_flops_total": 94632384,
            "reference": False,
        },
        {
            "id": "S-BETA-1",
            "activation": "softplus_beta",
            "beta": 1.0,
            "activation_flops_per_element": 5,
            "expected_flops_per_epoch": 850711121,
            "expected_inference_flops_total": 94632384,
            "reference": True,
        },
        {
            "id": "S-BETA-2",
            "activation": "softplus_beta",
            "beta": 2.0,
            "activation_flops_per_element": 5,
            "expected_flops_per_epoch": 850711121,
            "expected_inference_flops_total": 94632384,
            "reference": False,
        },
        {
            "id": "S-BETA-5",
            "activation": "softplus_beta",
            "beta": 5.0,
            "activation_flops_per_element": 5,
            "expected_flops_per_epoch": 850711121,
            "expected_inference_flops_total": 94632384,
            "reference": False,
        },
    ]
    if config["configurations"] != expected_configurations:
        raise ValueError("beta map differs from the frozen V2 protocol")


def _source_status() -> list[str]:
    watched = [
        "AGENTS.md",
        "Passo-a-passo.md",
        "requirements.txt",
        "bert_cpu",
        "datasets",
        "exercises/q01_activations.py",
        "exercises/task_binary_classification.py",
        "experiments/configs/v2_softplus_curvature.json",
        "experiments/hypotheses.md",
        "experiments/v2/protocol.md",
        "experiments/run_v1.py",
        "experiments/run_v2.py",
        "experiments/run_v2_all.py",
        "experiments/plot_v2.py",
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
            REPO_ROOT / "experiments/run_v2_all.py",
            REPO_ROOT / "experiments/plot_v2.py",
            config_path,
            REPO_ROOT / "experiments/hypotheses.md",
            REPO_ROOT / "experiments/v2/protocol.md",
            REPO_ROOT / "AGENTS.md",
            REPO_ROOT / "Passo-a-passo.md",
            REPO_ROOT / "requirements.txt",
        ]
    )
    return [path for path in paths if path.exists()]


def _read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
            raise ValueError("V2 results.csv header does not match runner schema")
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
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
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


def activation_diagnostics(
    model: adult.AdultMLP,
    X: np.ndarray,
    *,
    epoch: int,
) -> dict[str, Any]:
    """Summarize the V2 pre-activation, output and local derivative."""
    if model.activation != "softplus_beta" or model.activation_beta is None:
        raise ValueError("V2 diagnostics require a Softplus-beta model")
    weight = model.fc1.weight.data
    if X.ndim != 2 or X.shape[0] != weight.shape[0] - 1:
        raise ValueError("diagnostic input does not match fc1")

    z = weight[1:].T @ X + weight[0][:, None]
    beta = float(model.activation_beta)
    scaled = beta * z
    h = np.logaddexp(0.0, scaled) / beta
    derivative = common._stable_sigmoid(scaled)
    near_zero_threshold = 1e-6
    return {
        "event": "diagnostic",
        "epoch": epoch,
        "population": "validation",
        "samples": int(X.shape[1]),
        "values": int(z.size),
        "activation": model.activation,
        "beta": beta,
        "z": common._summary(z),
        "h": common._summary(h),
        "local_derivative": common._summary(derivative),
        "near_zero_h_fraction": float((np.abs(h) <= near_zero_threshold).mean()),
        "low_derivative_fraction": float(
            (derivative <= LOW_DERIVATIVE_THRESHOLD).mean()
        ),
        "high_derivative_fraction": float(
            (derivative >= HIGH_DERIVATIVE_THRESHOLD).mean()
        ),
        "transition_derivative_fraction": float(
            (
                (derivative > LOW_DERIVATIVE_THRESHOLD)
                & (derivative < HIGH_DERIVATIVE_THRESHOLD)
            ).mean()
        ),
    }


def result_artifacts_valid(
    row: dict[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> bool:
    """Validate one completed V2 row and its immutable evidence."""
    try:
        config_path = config_path.resolve()
        config = load_config(config_path)
        validate_frozen_config(config)
        selected = get_configuration(config, str(row["config_id"]))
        seed = int(row["seed"])
        repetition = int(row["repetition"])
        beta = float(row["beta"])
        expected_epoch_flops = int(selected["expected_flops_per_epoch"])
        expected_total_flops = expected_epoch_flops * int(config["training"]["epochs"])
        expected_inference_flops = int(selected["expected_inference_flops_total"])

        if (
            row["task"] != config["task"]
            or row["variable"] != config["variable"]
            or row["phase"] != config["phase"]
            or row["run_kind"] != "scientific"
            or row["purpose"] != "primary"
            or row["status"] != "completed_valid"
            or int(row["epochs"]) != int(config["training"]["epochs"])
            or str(row.get("test_accuracy", "")).strip()
            or row["activation"] != selected["activation"]
            or seed not in config["training"]["model_seeds"]
            or repetition != 1
            or row["run_id"] != expected_run_id(
                str(row["config_id"]),
                seed,
                repetition=repetition,
            )
            or beta != float(selected["beta"])
            or row["config_hash"] != common.sha256_file(config_path)
            or int(row["split_seed"]) != int(config["training"]["split_seed"])
            or row["split_hash"] != config["data"]["split_sha256"]
            or row["dataset_hash"] != config["data"]["encoded_train_sha256"]
            or row["initial_weights_hash"]
            != config["data"]["initial_weights_sha256"][str(seed)]
            or int(row["parameters"]) != int(config["architecture"]["parameters"])
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

        checkpoint_path = common._recorded_path(row["checkpoint_path"])
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive["metadata_json"].item()))
        if (
            manifest.get("variable") != row["variable"]
            or manifest.get("phase") != row["phase"]
            or manifest.get("purpose") != row["purpose"]
            or float(manifest.get("beta")) != beta
            or manifest.get("epoch") != int(config["training"]["epochs"])
            or manifest.get("architecture") != config["architecture"]
            or manifest.get("training") != config["training"]
            or manifest.get("configuration") != selected
        ):
            return False

        log_path = common._recorded_path(row["log_path"])
        events = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
        ]
        started = [event for event in events if event.get("event") == "run_started"]
        epochs = [event for event in events if event.get("event") == "epoch"]
        diagnostics = [
            event for event in events if event.get("event") == "diagnostic"
        ]
        if len(started) != 1:
            return False
        start = started[0]
        if (
            start.get("configuration") != selected
            or start.get("beta") != selected["beta"]
            or start.get("data", {}).get("split_loaded") != "train"
            or start.get("data", {}).get("official_test_loaded") is not False
            or start.get("training", {}).get("evaluate_test") is not False
            or start.get("training", {}).get("epochs")
            != int(config["training"]["epochs"])
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
                for event in epochs
                for field in metric_fields
            ]
        )
        final = epochs[-1]
        return (
            [event["epoch"] for event in epochs]
            == list(range(1, int(config["training"]["epochs"]) + 1))
            and all(int(event["flops"]) == expected_epoch_flops for event in epochs)
            and np.isfinite(metric_values).all()
            and float(row["train_loss_final"]) == float(final["train_loss"])
            and float(row["val_loss_final"]) == float(final["val_loss"])
            and float(row["train_accuracy"]) == float(final["train_accuracy"])
            and float(row["val_accuracy"]) == float(final["val_accuracy"])
            and [event["epoch"] for event in diagnostics]
            == list(DIAGNOSTIC_EPOCHS)
            and all(
                float(event["beta"]) == beta
                and event.get("population") == "validation"
                and int(event["samples"]) == 6_512
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


def _reference_complete(
    rows: list[dict[str, str]],
    *,
    verify_artifacts: bool,
    config_path: Path = DEFAULT_CONFIG,
) -> bool:
    references = [
        row
        for row in rows
        if row["config_id"] == REFERENCE_CONFIG_ID
        and row["status"] == "completed_valid"
        and (
            not verify_artifacts
            or result_artifacts_valid(row, config_path=config_path)
        )
    ]
    return (
        len(references) == 3
        and {int(row["seed"]) for row in references} == {0, 1, 2}
        and all(int(row["repetition"]) == 1 for row in references)
    )


def _reference_matches_context(
    rows: list[dict[str, str]],
    expected: dict[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> bool:
    references = [row for row in rows if row["config_id"] == REFERENCE_CONFIG_ID]
    return _reference_complete(
        references,
        verify_artifacts=True,
        config_path=config_path,
    ) and all(
        all(row[field] == str(value) for field, value in expected.items())
        for row in references
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
    """Execute one isolated V2 configuration and persist its evidence."""
    config_path = config_path.resolve()
    artifacts_dir = artifacts_dir.resolve()
    config = load_config(config_path)
    validate_frozen_config(config)
    selected = get_configuration(config, config_id)
    training_config = config["training"]

    if seed not in training_config["model_seeds"]:
        raise ValueError(f"seed {seed} is not in the frozen configuration")
    if repetition != 1:
        raise ValueError("V2 permits exactly one repetition per seed")

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
    if (
        not smoke
        and config_id != REFERENCE_CONFIG_ID
        and not _reference_complete(
            existing_results,
            verify_artifacts=True,
            config_path=config_path,
        )
    ):
        raise RuntimeError(
            "V2 variants are blocked until S-BETA-1 has all three valid seeds"
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
        "activation": selected["activation"],
        "beta": selected["beta"],
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
        model = adult.build_model(
            train_dataset.n_features,
            activation=selected["activation"],
            model_seed=seed,
            activation_beta=float(selected["beta"]),
        )
        parameters = adult.parameter_count(model)
        if parameters != int(config["architecture"]["parameters"]):
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
        if initial_weights_hash != frozen_data["initial_weights_sha256"][str(seed)]:
            raise RuntimeError("initial weights hash differs from frozen configuration")

        majority_accuracy = float(np.bincount(y_val).max() / len(y_val))
        context = {
            "task": config["task"],
            "variable": config["variable"],
            "run_kind": "scientific",
            "phase": config["phase"],
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
            "parameters": parameters,
        }
        if (
            not smoke
            and config_id != REFERENCE_CONFIG_ID
            and not _reference_matches_context(
                existing_results,
                context,
                config_path=config_path,
            )
        ):
            raise RuntimeError(
                "S-BETA-1 artifacts do not match the current V2 context"
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
            "activation": selected["activation"],
            "beta": selected["beta"],
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
                "architecture": config["architecture"],
                "parameters": parameters,
                "initial_weights_hash": initial_weights_hash,
                "activation_beta": selected["beta"],
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

        initial_flops = cpu.flop_count()
        initial_parameter_hash = common.parameter_hash(model)
        initial_rng = np.random.get_state()
        common._append_jsonl(
            log_path,
            activation_diagnostics(model, X_val, epoch=0),
        )
        if cpu.flop_count() != initial_flops:
            raise RuntimeError("epoch-0 diagnostics changed the FLOP counter")
        if common.parameter_hash(model) != initial_parameter_hash:
            raise RuntimeError("epoch-0 diagnostics changed model parameters")
        if not common._rng_state_equal(initial_rng, np.random.get_state()):
            raise RuntimeError("epoch-0 diagnostics changed RNG state")
        cpu.reset_flops()

        def record_epoch(
            metrics: adult.EpochMetrics,
            current_model: adult.AdultMLP,
        ) -> None:
            common._append_jsonl(log_path, {"event": "epoch", **asdict(metrics)})
            if metrics.epoch not in DIAGNOSTIC_EPOCHS:
                return
            flops_before = cpu.flop_count()
            weights_before = common.parameter_hash(current_model)
            rng_before = np.random.get_state()
            diagnostic = activation_diagnostics(
                current_model,
                X_val,
                epoch=metrics.epoch,
            )
            if cpu.flop_count() != flops_before:
                raise RuntimeError("diagnostics changed the FLOP counter")
            if common.parameter_hash(current_model) != weights_before:
                raise RuntimeError("diagnostics changed model parameters")
            if not common._rng_state_equal(rng_before, np.random.get_state()):
                raise RuntimeError("diagnostics changed RNG state")
            common._append_jsonl(log_path, diagnostic)
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
            "activation": selected["activation"],
            "beta": selected["beta"],
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
            "architecture": config["architecture"],
            "training": training_config,
            "configuration": selected,
        }
        checkpoint_hash = common.save_checkpoint(
            checkpoint_path,
            model,
            checkpoint_manifest,
        )
        verification_model = adult.build_model(
            train_dataset.n_features,
            activation=selected["activation"],
            model_seed=seed,
            activation_beta=float(selected["beta"]),
        )
        loaded_manifest = common.load_checkpoint(
            checkpoint_path,
            verification_model,
            expected_sha256=checkpoint_hash,
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
            "activation": selected["activation"],
            "beta": selected["beta"],
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
        description="Run one frozen Variable 2 Softplus-beta configuration.",
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
    command = [sys.executable, "-m", "experiments.run_v2", *(argv or sys.argv[1:])]
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
        f"beta={float(result['beta']):g} | "
        f"val={result['val_accuracy']:.4f} | FLOPs={result['flops_total']:,}"
    )
    return result


if __name__ == "__main__":
    main()
