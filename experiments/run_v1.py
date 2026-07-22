"""Reproducible runner for Variable 1 of the Adult/q01 study.

Scientific runs always use the frozen JSON configuration and never load the
official Adult test split. ``--smoke`` uses two epochs, writes isolated debug
artifacts and never appends to ``results.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import datasets  # noqa: E402
from bert_cpu import engine as cpu  # noqa: E402
from exercises import task_binary_classification as adult  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/v1_activation_family.json"
DEFAULT_ARTIFACTS = REPO_ROOT / "experiments"
DIAGNOSTIC_EPOCHS = (0, 1, 25, 50, 75, 100)
DIAGNOSTIC_PERCENTILES = (1, 5, 50, 95, 99)
NEAR_ZERO_THRESHOLD = 1e-6
LOW_DERIVATIVE_THRESHOLD = 0.05
EXPECTED_BASE_COMMIT = "14e65c20c2312e66ba76e54431b396f60ce65e10"

RESULT_FIELDS = (
    "run_id",
    "task",
    "variable",
    "config_id",
    "activation",
    "seed",
    "repetition",
    "run_kind",
    "phase",
    "purpose",
    "status",
    "branch",
    "commit",
    "base_commit",
    "code_state_hash",
    "config_hash",
    "environment_hash",
    "dataset_hash",
    "split_seed",
    "split_hash",
    "initial_weights_hash",
    "final_weights_hash",
    "epochs",
    "optimizer",
    "learning_rate",
    "train_samples",
    "val_samples",
    "parameters",
    "train_loss_final",
    "val_loss_final",
    "train_accuracy",
    "val_accuracy",
    "test_accuracy",
    "flops_per_epoch",
    "flops_total",
    "gflops_total",
    "inference_flops_total",
    "inference_flops_per_sample",
    "checkpoint_path",
    "checkpoint_hash",
    "log_path",
    "notes",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """SHA-256 of one file, streamed to avoid unnecessary copies."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_arrays(*arrays: np.ndarray) -> str:
    """Canonical array hash already used by this study's recorded hashes."""
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(repr(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def parameter_hash(model: adult.AdultMLP) -> str:
    return hash_arrays(*(parameter.data for parameter in model.parameters()))


def _hash_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}):
        relative = path.relative_to(REPO_ROOT).as_posix()
        data = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_files(config_path: Path) -> list[Path]:
    paths = list((REPO_ROOT / "bert_cpu").glob("*.py"))
    paths.extend(
        [
            REPO_ROOT / "datasets/__init__.py",
            REPO_ROOT / "datasets/loaders.py",
            REPO_ROOT / "exercises/q01_activations.py",
            REPO_ROOT / "exercises/task_binary_classification.py",
            Path(__file__),
            config_path,
            REPO_ROOT / "experiments/hypotheses.md",
            REPO_ROOT / "AGENTS.md",
            REPO_ROOT / "Passo-a-passo.md",
            REPO_ROOT / "requirements.txt",
        ]
    )
    return [path for path in paths if path.exists()]


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _environment_metadata() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "default_dtype": str(cpu.default_dtype),
        "requirements_hash": sha256_file(REPO_ROOT / "requirements.txt"),
    }


def _source_status() -> list[str]:
    watched = [
        "AGENTS.md",
        "Passo-a-passo.md",
        "requirements.txt",
        "bert_cpu",
        "datasets",
        "exercises/q01_activations.py",
        "exercises/task_binary_classification.py",
        "experiments/configs",
        "experiments/hypotheses.md",
        "experiments/run_v1.py",
        "test",
    ]
    output = _git("status", "--porcelain", "--untracked-files=all", "--", *watched)
    return output.splitlines() if output else []


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"task", "variable", "base_commit", "architecture", "training", "configurations"}
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
    """Fail if the runner and the pre-experimental JSON have drifted apart."""
    if (
        config.get("schema_version") != 1
        or config.get("created_at") != "2026-07-21"
        or config.get("task") != "adult_binary_classification"
        or config.get("variable") != "V1_activation_family"
        or config.get("branch") != "q01-ativacoes-adult"
        or config.get("base_commit") != EXPECTED_BASE_COMMIT
        or config.get("architecture")
        != {
            "input_features": 108,
            "hidden_features": 64,
            "output_classes": 2,
            "parameters": 7106,
        }
    ):
        raise ValueError("identity or architecture differs from the frozen V1 protocol")

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
    if config.get("data") != expected_data:
        raise ValueError("data hashes differ from the frozen V1 protocol")

    training = config["training"]
    if (
        training["batching"] != "full_batch"
        or training["optimizer"] != "Adam"
        or float(training["learning_rate"]) != 1e-2
        or int(training["epochs"]) != 100
        or float(training["validation_fraction"]) != 0.2
        or int(training["split_seed"]) != 0
        or list(training["model_seeds"]) != [0, 1, 2]
    ):
        raise ValueError("training configuration differs from the frozen V1 protocol")

    diagnostics = config.get("diagnostics", {})
    expected_diagnostics = {
        "population": "fixed_validation_split",
        "epochs": list(DIAGNOSTIC_EPOCHS),
        "percentiles": list(DIAGNOSTIC_PERCENTILES),
        "near_zero_threshold": NEAR_ZERO_THRESHOLD,
        "low_abs_derivative_threshold": LOW_DERIVATIVE_THRESHOLD,
        "sigmoid_saturation_bounds": [0.05, 0.95],
    }
    if diagnostics != expected_diagnostics:
        raise ValueError("diagnostic configuration differs from the V1 runner")

    expected_evaluation = {
        "primary_metric": "mean_validation_accuracy_at_epoch_100",
        "relevant_difference_percentage_points": 0.5,
        "required_seed_agreement": 2,
        "checkpoint": "epoch_100",
        "test_policy": "final_phase_only_all_valid_configurations",
    }
    if config.get("evaluation") != expected_evaluation:
        raise ValueError("evaluation differs from the frozen V1 protocol")

    expected_artifacts = {
        "log_format": "jsonl",
        "checkpoint_format": "npz_without_pickle",
        "smoke_appends_results": False,
        "scientific_run_requires_clean_source": True,
    }
    if config.get("artifacts") != expected_artifacts:
        raise ValueError("artifact policy differs from the V1 runner")

    expected_configurations = [
        {
            "id": "F-RELU",
            "activation": "relu",
            "activation_flops_per_element": 1,
            "expected_flops_per_epoch": 842375505,
            "expected_inference_flops_total": 92965312,
            "baseline": True,
        },
        {
            "id": "F-SIGMOID",
            "activation": "sigmoid",
            "activation_flops_per_element": 4,
            "expected_flops_per_epoch": 848627217,
            "expected_inference_flops_total": 94215616,
            "baseline": False,
        },
        {
            "id": "F-SWISH",
            "activation": "swish",
            "activation_flops_per_element": 5,
            "expected_flops_per_epoch": 850711121,
            "expected_inference_flops_total": 94632384,
            "baseline": False,
        },
        {
            "id": "F-SOFTPLUS",
            "activation": "softplus",
            "activation_flops_per_element": 3,
            "expected_flops_per_epoch": 846543313,
            "expected_inference_flops_total": 93798848,
            "baseline": False,
        },
    ]
    if config["configurations"] != expected_configurations:
        raise ValueError("activation map differs from the frozen V1 protocol")


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    non_negative = values >= 0.0
    result[non_negative] = 1.0 / (1.0 + np.exp(-values[non_negative]))
    exp_values = np.exp(values[~non_negative])
    result[~non_negative] = exp_values / (1.0 + exp_values)
    return result


def _summary(values: np.ndarray) -> dict[str, float]:
    if not np.isfinite(values).all():
        raise FloatingPointError("diagnostic contains NaN or Inf")
    result = {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    result.update(
        {
            f"p{percentile:02d}": float(np.percentile(values, percentile))
            for percentile in DIAGNOSTIC_PERCENTILES
        }
    )
    return result


def activation_diagnostics(
    model: adult.AdultMLP,
    X: np.ndarray,
    *,
    epoch: int,
) -> dict[str, Any]:
    """Summarise ``z``, ``h`` and the local derivative using NumPy only."""
    weight = model.fc1.weight.data
    if X.ndim != 2 or X.shape[0] != weight.shape[0] - 1:
        raise ValueError("diagnostic input does not match fc1")

    z = weight[1:].T @ X + weight[0][:, None]
    sigmoid = _stable_sigmoid(z)
    if model.activation == "relu":
        h = np.maximum(0.0, z)
        derivative = (z > 0.0).astype(z.dtype)
    elif model.activation == "sigmoid":
        h = sigmoid
        derivative = sigmoid * (1.0 - sigmoid)
    elif model.activation == "swish":
        h = z * sigmoid
        derivative = sigmoid + z * sigmoid * (1.0 - sigmoid)
    elif model.activation == "softplus":
        h = np.logaddexp(0.0, z)
        derivative = sigmoid
    else:  # AdultMLP rejects this earlier; keep diagnostics defensive.
        raise ValueError(f"unsupported activation: {model.activation!r}")

    diagnostic: dict[str, Any] = {
        "event": "diagnostic",
        "epoch": epoch,
        "population": "validation",
        "samples": int(X.shape[1]),
        "values": int(z.size),
        "activation": model.activation,
        "z": _summary(z),
        "h": _summary(h),
        "local_derivative": _summary(derivative),
        "near_zero_h_fraction": float((np.abs(h) <= NEAR_ZERO_THRESHOLD).mean()),
        "low_abs_derivative_fraction": float(
            (np.abs(derivative) <= LOW_DERIVATIVE_THRESHOLD).mean()
        ),
    }
    if model.activation == "relu":
        diagnostic["exact_zero_h_fraction"] = float((h == 0.0).mean())
    if model.activation == "sigmoid":
        diagnostic["sigmoid_saturation_fraction"] = float(
            ((h <= 0.05) | (h >= 0.95)).mean()
        )
    return diagnostic


def _rng_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reserve_run(artifacts_dir: Path, run_id: str) -> Path:
    """Atomically reserve a run id; failed attempts keep their reservation."""
    directory = artifacts_dir / ".run_reservations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.json"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"run id is already reserved: {run_id}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {"run_id": run_id, "reserved_at_utc": _utc_now(), "pid": os.getpid()},
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(directory)
    return path


def _read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
            raise ValueError("results.csv header does not match runner schema")
        return list(reader)


def _append_result(path: Path, row: dict[str, Any]) -> None:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = _read_results(path)
        if any(existing["run_id"] == row["run_id"] for existing in rows):
            raise FileExistsError(f"run already recorded: {row['run_id']}")
        rows.append({field: row.get(field, "") for field in RESULT_FIELDS})

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent, text=True
        )
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
            _fsync_directory(path.parent)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)


def _remove_result(path: Path, run_id: str) -> None:
    """Best-effort rollback if the terminal log write fails after CSV append."""
    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows = _read_results(path)
        remaining = [row for row in rows if row["run_id"] != run_id]
        if len(remaining) == len(rows):
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent, text=True
        )
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
                writer.writeheader()
                writer.writerows(remaining)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
            _fsync_directory(path.parent)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)


def save_checkpoint(
    path: Path,
    model: adult.AdultMLP,
    metadata: dict[str, Any],
) -> str:
    """Atomically save named weights plus a JSON manifest, without pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names_and_parameters = list(model.named_parameters())
    manifest = dict(metadata)
    manifest["parameter_names"] = [name for name, _ in names_and_parameters]
    manifest["parameter_shapes"] = {
        name: list(parameter.shape) for name, parameter in names_and_parameters
    }
    arrays = {
        f"parameter_{index}": parameter.data
        for index, (_, parameter) in enumerate(names_and_parameters)
    }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary_name,
            metadata_json=np.array(json.dumps(manifest, sort_keys=True)),
            **arrays,
        )
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return sha256_file(path)


def load_checkpoint(
    path: Path,
    model: adult.AdultMLP,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a checkpoint completely, then restore it without partial writes."""
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    names_and_parameters = list(model.named_parameters())
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["metadata_json"].item()))
        expected_names = [name for name, _ in names_and_parameters]
        if manifest["parameter_names"] != expected_names:
            raise ValueError("checkpoint parameter names do not match model")
        if manifest.get("activation") != model.activation:
            raise ValueError("checkpoint activation does not match model")
        architecture = manifest.get("architecture", {})
        if (
            architecture.get("input_features") != model.n_features
            or architecture.get("hidden_features") != model.hidden
            or architecture.get("output_classes") != 2
        ):
            raise ValueError("checkpoint architecture does not match model")

        stored_parameters: list[np.ndarray] = []
        for index, (_, parameter) in enumerate(names_and_parameters):
            stored = archive[f"parameter_{index}"].copy()
            if stored.shape != parameter.shape or stored.dtype != parameter.data.dtype:
                raise ValueError("checkpoint parameter shape or dtype mismatch")
            if not np.isfinite(stored).all():
                raise ValueError("checkpoint contains NaN or Inf")
            stored_parameters.append(stored)
    if hash_arrays(*stored_parameters) != manifest.get("final_weights_hash"):
        raise ValueError("checkpoint final-weights hash mismatch")
    for stored, (_, parameter) in zip(stored_parameters, names_and_parameters):
        parameter.data[...] = stored
    return manifest


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _result_artifacts_valid(row: dict[str, str]) -> bool:
    """Verify the immutable evidence referenced by one completed CSV row."""
    try:
        log_path = _recorded_path(row["log_path"])
        checkpoint_path = _recorded_path(row["checkpoint_path"])
        if not log_path.is_file() or not checkpoint_path.is_file():
            return False
        if sha256_file(checkpoint_path) != row["checkpoint_hash"]:
            return False

        events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        if not events or events[-1].get("event") != "run_completed":
            return False
        if events[-1].get("status") != row["status"]:
            return False
        if row["run_kind"] == "scientific" and not events[-1].get("result_registered"):
            return False
        if not any(
            event.get("event") == "run_attempt_started"
            and event.get("run_id") == row["run_id"]
            for event in events
        ):
            return False

        with np.load(checkpoint_path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive["metadata_json"].item()))
            parameter_names = manifest["parameter_names"]
            stored = [archive[f"parameter_{index}"].copy() for index in range(len(parameter_names))]
        if not all(np.isfinite(array).all() for array in stored):
            return False
        if hash_arrays(*stored) != row["final_weights_hash"]:
            return False
        expected_manifest = {
            "run_id": row["run_id"],
            "run_kind": row["run_kind"],
            "config_id": row["config_id"],
            "activation": row["activation"],
            "seed": int(row["seed"]),
            "repetition": int(row["repetition"]),
            "split_hash": row["split_hash"],
            "config_hash": row["config_hash"],
            "environment_hash": row["environment_hash"],
            "commit": row["commit"],
            "code_state_hash": row["code_state_hash"],
            "initial_weights_hash": row["initial_weights_hash"],
            "final_weights_hash": row["final_weights_hash"],
        }
        return all(manifest.get(key) == value for key, value in expected_manifest.items())
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _baseline_complete(
    results: list[dict[str, str]],
    *,
    verify_artifacts: bool = False,
) -> bool:
    observed = [
        row
        for row in results
        if row["config_id"] == "F-RELU"
        and row["status"] == "completed_valid"
        and (not verify_artifacts or _result_artifacts_valid(row))
    ]
    seed_zero_repetitions = {
        int(row["repetition"]) for row in observed if int(row["seed"]) == 0
    }
    return (
        len(seed_zero_repetitions) >= 2
        and any(int(row["seed"]) == 1 for row in observed)
        and any(int(row["seed"]) == 2 for row in observed)
    )


def _baseline_matches_context(
    results: list[dict[str, str]],
    expected: dict[str, Any],
    *,
    verify_artifacts: bool = False,
) -> bool:
    baseline_rows = [
        row
        for row in results
        if row["config_id"] == "F-RELU" and row["status"] == "completed_valid"
    ]
    if not _baseline_complete(baseline_rows, verify_artifacts=verify_artifacts):
        return False
    return all(
        all(row[field] == str(value) for field, value in expected.items())
        for row in baseline_rows
    )


def _relu_reference(
    results: list[dict[str, str]],
    *,
    verify_artifacts: bool = False,
) -> dict[str, str] | None:
    candidates = [
        row
        for row in results
        if row["config_id"] == "F-RELU"
        and row["seed"] == "0"
        and row["status"] == "completed_valid"
        and (not verify_artifacts or _result_artifacts_valid(row))
    ]
    return min(candidates, key=lambda row: int(row["repetition"]), default=None)


def _run_purpose(
    *,
    config_id: str,
    seed: int,
    smoke: bool,
    existing_results: list[dict[str, str]],
) -> str:
    if smoke:
        return "smoke_validation"
    valid_same_seed = [
        row
        for row in existing_results
        if row["config_id"] == config_id
        and int(row["seed"]) == seed
        and row["status"] == "completed_valid"
        and _result_artifacts_valid(row)
    ]
    if config_id == "F-RELU" and seed == 0:
        if len(valid_same_seed) >= 2:
            raise RuntimeError("ReLU seed 0 already has a reference and determinism repeat")
        return "determinism_repeat" if valid_same_seed else "primary"
    if valid_same_seed:
        raise RuntimeError(f"a valid primary run already exists for {config_id} seed {seed}")
    return "primary"


def validate_relu_reproduction(
    row: dict[str, Any],
    existing_results: list[dict[str, str]],
    *,
    verify_artifacts: bool = False,
) -> None:
    """Require an additional seed-0 ReLU run to reproduce the reference."""
    if row["config_id"] != "F-RELU" or row["seed"] != 0:
        return
    reference = _relu_reference(existing_results, verify_artifacts=verify_artifacts)
    if reference is None:
        return

    exact_fields = (
        "commit",
        "code_state_hash",
        "config_hash",
        "dataset_hash",
        "split_hash",
        "initial_weights_hash",
        "final_weights_hash",
        "parameters",
        "flops_per_epoch",
        "flops_total",
        "inference_flops_total",
    )
    mismatches = [field for field in exact_fields if str(row[field]) != reference[field]]
    numeric_fields = (
        "train_loss_final",
        "val_loss_final",
        "train_accuracy",
        "val_accuracy",
        "inference_flops_per_sample",
    )
    mismatches.extend(
        field
        for field in numeric_fields
        if not np.isclose(
            float(row[field]),
            float(reference[field]),
            rtol=1e-12,
            atol=1e-12,
        )
    )
    if mismatches:
        raise RuntimeError(f"ReLU reproduction mismatch: {sorted(set(mismatches))}")


def _validate_history(
    training: adult.TrainingResult,
    epochs: int,
    expected_flops_per_epoch: int,
) -> int:
    if len(training.history) != epochs:
        raise RuntimeError("training history has the wrong number of epochs")
    expected = list(range(1, epochs + 1))
    if [item.epoch for item in training.history] != expected:
        raise RuntimeError("training history has missing or reordered epochs")
    values = np.array(
        [
            [item.train_loss, item.val_loss, item.train_accuracy, item.val_accuracy]
            for item in training.history
        ]
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("training history contains NaN or Inf")
    flops = [item.flops for item in training.history]
    if training.total_flops != sum(flops):
        raise RuntimeError("total FLOPs do not equal the epoch sum")
    if len(set(flops)) != 1:
        raise RuntimeError("FLOPs changed between epochs of the same run")
    if flops[0] != expected_flops_per_epoch:
        raise RuntimeError(
            f"epoch FLOPs differ from frozen value: {flops[0]} != {expected_flops_per_epoch}"
        )
    return flops[0]


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
    """Execute one isolated V1 run and persist its evidence."""
    config_path = config_path.resolve()
    artifacts_dir = artifacts_dir.resolve()
    config = load_config(config_path)
    validate_frozen_config(config)
    selected = get_configuration(config, config_id)
    training_config = config["training"]

    if seed not in training_config["model_seeds"]:
        raise ValueError(f"seed {seed} is not in the frozen configuration")
    if repetition <= 0:
        raise ValueError("repetition must be positive")

    epochs = 2 if smoke else int(training_config["epochs"])
    run_kind = "smoke" if smoke else "scientific"
    prefix = "SMOKE-" if smoke else ""
    run_id = f"{prefix}{config_id}-s{seed}-r{repetition}"
    subdirectory = "smoke" if smoke else ""
    log_path = artifacts_dir / "logs" / subdirectory / f"{run_id}.jsonl"
    checkpoint_path = artifacts_dir / "checkpoints" / subdirectory / f"{run_id}.npz"
    results_path = artifacts_dir / "results.csv"

    if log_path.exists() or checkpoint_path.exists():
        raise FileExistsError(f"artifacts already exist for {run_id}")
    existing_results = _read_results(results_path) if results_path.exists() else []
    if any(row["run_id"] == run_id for row in existing_results):
        raise FileExistsError(f"run already exists in results.csv: {run_id}")
    purpose = _run_purpose(
        config_id=config_id,
        seed=seed,
        smoke=smoke,
        existing_results=existing_results,
    )
    if (
        not smoke
        and config_id != "F-RELU"
        and not _baseline_complete(existing_results, verify_artifacts=True)
    ):
        raise RuntimeError("V1 variants are blocked until the ReLU baseline is complete")

    branch = _git("branch", "--show-current")
    commit = _git("rev-parse", "HEAD")
    source_status = _source_status()
    if not smoke and branch != config["branch"]:
        raise RuntimeError(f"scientific run requires branch {config['branch']!r}")
    if not smoke and source_status:
        raise RuntimeError(f"scientific run requires clean source state: {source_status}")
    source_files = _source_files(config_path)
    code_state_hash = _hash_files(source_files)
    config_hash = sha256_file(config_path)
    environment = _environment_metadata()
    environment_hash = _hash_json(environment)
    reservation_path = _reserve_run(artifacts_dir, run_id)

    start_event: dict[str, Any] | None = {
        "event": "run_attempt_started",
        "schema_version": 1,
        "timestamp_utc": _utc_now(),
        "run_id": run_id,
        "run_kind": run_kind,
        "phase": "v1_train_validation",
        "purpose": purpose,
        "status": "initializing",
        "config_id": config_id,
        "activation": selected["activation"],
        "seed": seed,
        "repetition": repetition,
        "command": list(command or sys.argv),
        "branch": branch,
        "commit": commit,
        "code_state_hash": code_state_hash,
        "config_hash": config_hash,
        "environment_hash": environment_hash,
        "source_status": source_status,
        "reservation_path": _display_path(reservation_path),
    }
    _append_jsonl(log_path, start_event)
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

        expected_features = int(config["architecture"]["input_features"])
        if train_dataset.n_features != expected_features:
            raise RuntimeError("Adult feature count differs from frozen configuration")

        model = adult.build_model(
            train_dataset.n_features,
            activation=selected["activation"],
            model_seed=seed,
        )
        parameters = adult.parameter_count(model)
        if parameters != int(config["architecture"]["parameters"]):
            raise RuntimeError("model parameter count differs from frozen configuration")

        dataset_hash = hash_arrays(train_dataset.X, train_dataset.y)
        split_hash = hash_arrays(train_indices, val_indices)
        initial_weights_hash = parameter_hash(model)
        raw_train_path = REPO_ROOT / "datasets/adult/adult.data"
        raw_train_hash = sha256_file(raw_train_path)
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

        if config_id != "F-RELU" and not smoke:
            baseline_context = {
                "task": config["task"],
                "variable": config["variable"],
                "run_kind": "scientific",
                "phase": "v1_train_validation",
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
            if not _baseline_matches_context(
                existing_results,
                baseline_context,
                verify_artifacts=True,
            ):
                raise RuntimeError("ReLU baseline artifacts do not match the current context")

        start_event = {
            "event": "run_started",
            "schema_version": 1,
            "timestamp_utc": _utc_now(),
            "run_id": run_id,
            "run_kind": run_kind,
            "phase": "v1_train_validation",
            "purpose": purpose,
            "status": "running",
            "task": config["task"],
            "variable": config["variable"],
            "config_id": config_id,
            "activation": selected["activation"],
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
                "config_path": _display_path(config_path),
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
                "preprocessing_limitation": "encoder_fit_on_official_train_before_holdout",
            },
            "model": {
                "architecture": config["architecture"],
                "parameters": parameters,
                "initial_weights_hash": initial_weights_hash,
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
            "diagnostics": {
                "population": "fixed_validation_split",
                "epochs": list(DIAGNOSTIC_EPOCHS),
                "percentiles": list(DIAGNOSTIC_PERCENTILES),
                "near_zero_threshold": NEAR_ZERO_THRESHOLD,
                "low_abs_derivative_threshold": LOW_DERIVATIVE_THRESHOLD,
                "sigmoid_saturation_bounds": [0.05, 0.95],
            },
        }
        _append_jsonl(log_path, start_event)

        initial_flops = cpu.flop_count()
        initial_parameter_hash = parameter_hash(model)
        initial_rng = np.random.get_state()
        _append_jsonl(log_path, activation_diagnostics(model, X_val, epoch=0))
        if cpu.flop_count() != initial_flops:
            raise RuntimeError("epoch-0 diagnostics changed the FLOP counter")
        if parameter_hash(model) != initial_parameter_hash:
            raise RuntimeError("epoch-0 diagnostics changed model parameters")
        if not _rng_state_equal(initial_rng, np.random.get_state()):
            raise RuntimeError("epoch-0 diagnostics changed RNG state")
        cpu.reset_flops()

        def record_epoch(metrics: adult.EpochMetrics, current_model: adult.AdultMLP) -> None:
            _append_jsonl(log_path, {"event": "epoch", **asdict(metrics)})
            if metrics.epoch not in DIAGNOSTIC_EPOCHS:
                return
            flops_before = cpu.flop_count()
            weights_before = parameter_hash(current_model)
            rng_before = np.random.get_state()
            diagnostic = activation_diagnostics(current_model, X_val, epoch=metrics.epoch)
            if cpu.flop_count() != flops_before:
                raise RuntimeError("diagnostics changed the FLOP counter")
            if parameter_hash(current_model) != weights_before:
                raise RuntimeError("diagnostics changed model parameters")
            if not _rng_state_equal(rng_before, np.random.get_state()):
                raise RuntimeError("diagnostics changed RNG state")
            _append_jsonl(log_path, diagnostic)
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
        flops_per_epoch = _validate_history(
            training,
            epochs,
            int(selected["expected_flops_per_epoch"]),
        )
        if not all(np.isfinite(parameter.data).all() for parameter in model.parameters()):
            raise FloatingPointError("final weights contain NaN or Inf")

        final_weights_hash = parameter_hash(model)
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
            "epoch": epochs,
            "config_id": config_id,
            "activation": selected["activation"],
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
        checkpoint_hash = save_checkpoint(checkpoint_path, model, checkpoint_manifest)

        verification_model = adult.build_model(
            train_dataset.n_features,
            activation=selected["activation"],
            model_seed=seed,
        )
        loaded_manifest = load_checkpoint(
            checkpoint_path,
            verification_model,
            expected_sha256=checkpoint_hash,
        )
        if loaded_manifest != {
            **checkpoint_manifest,
            "parameter_names": [name for name, _ in model.named_parameters()],
            "parameter_shapes": {
                name: list(parameter.shape) for name, parameter in model.named_parameters()
            },
        }:
            raise RuntimeError("checkpoint manifest round-trip failed")
        if parameter_hash(verification_model) != final_weights_hash:
            raise RuntimeError("checkpoint weights round-trip failed")

        final_metrics = training.history[-1]
        terminal_status = "smoke_passed" if smoke else "completed_valid"
        row = {
            "run_id": run_id,
            "task": config["task"],
            "variable": config["variable"],
            "config_id": config_id,
            "activation": selected["activation"],
            "seed": seed,
            "repetition": repetition,
            "run_kind": run_kind,
            "phase": "v1_train_validation",
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
            "checkpoint_path": _display_path(checkpoint_path),
            "checkpoint_hash": checkpoint_hash,
            "log_path": _display_path(log_path),
            "notes": "official_test_not_loaded; smoke_not_scientific"
            if smoke
            else "official_test_not_loaded",
        }
        if not smoke:
            validate_relu_reproduction(
                row,
                existing_results,
                verify_artifacts=True,
            )
        artifacts_event = {
            "event": "artifacts_verified",
            "timestamp_utc": _utc_now(),
            "status": "artifacts_verified",
            "summary": row,
            "checkpoint_verified": True,
        }
        _append_jsonl(log_path, artifacts_event)

        if not smoke:
            _append_result(results_path, row)
        terminal_event = {
            "event": "run_completed",
            "timestamp_utc": _utc_now(),
            "status": terminal_status,
            "result_registered": not smoke,
        }
        try:
            _append_jsonl(log_path, terminal_event)
        except Exception:
            if not smoke:
                _remove_result(results_path, run_id)
            raise
        return row
    except Exception as error:
        if start_event is not None:
            _append_jsonl(
                log_path,
                {
                    "event": "run_failed",
                    "timestamp_utc": _utc_now(),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one frozen Variable 1 configuration.")
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
    command = [sys.executable, "-m", "experiments.run_v1", *(argv or sys.argv[1:])]
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
        f"val={result['val_accuracy']:.4f} | FLOPs={result['flops_total']:,}"
    )
    return result


if __name__ == "__main__":
    main()
