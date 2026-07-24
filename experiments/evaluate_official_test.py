"""Evaluate every frozen Adult/q01 checkpoint on the official test split.

This module is deliberately separate from the three training runners.  It
never trains, never accepts a configuration subset and never changes the
training/validation CSVs.  The workflow has two explicit commands:

1. ``--preflight`` validates the complete 33-checkpoint grid using only the
   training file and reproduces every recorded validation accuracy/FLOP count.
2. ``--evaluate-official-test`` repeats that preflight, loads Adult test once
   and evaluates every checkpoint with one forward pass.

The final CSV is published only after all 33 evaluations succeed.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

import datasets  # noqa: E402
from bert_cpu import engine as cpu  # noqa: E402
from exercises import task_binary_classification as adult  # noqa: E402
from experiments import plot_v1, plot_v2, plot_v3  # noqa: E402
from experiments import run_v1, run_v2, run_v3  # noqa: E402


DEFAULT_ARTIFACTS = REPO_ROOT / "experiments/final_evaluation"
INPUT_MANIFEST_NAME = "input_manifest.json"
LOG_NAME = "evaluation.jsonl"
RESULTS_NAME = "results.csv"
OUTPUT_MANIFEST_NAME = "output_manifest.json"
LOCK_NAME = ".evaluation.lock"

PHASE = "official_test_evaluation"
EXPECTED_TEST_FEATURES = 108
EXPECTED_TEST_SAMPLES = 16_281
EXPECTED_VALIDATION_SAMPLES = 6_512
EXPECTED_RAW_TRAIN_SHA256 = (
    "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d"
)
EXPECTED_RAW_TEST_SHA256 = (
    "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"
)
EXPECTED_RAW_TEST_GIT_BLOB = "b67e1fbb62332b2222482be2a0105e496aa281a8"
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_CONFIG_ORDER = (
    "F-RELU",
    "F-SIGMOID",
    "F-SWISH",
    "F-SOFTPLUS",
    "S-BETA-0.5",
    "S-BETA-1",
    "S-BETA-2",
    "S-BETA-5",
    "L1-DIRECT",
    "L2-IDENTITY",
    "L3-IDENTITY",
)
EXPECTED_RUN_IDS = tuple(
    f"{config_id}-s{seed}-r1"
    for config_id in EXPECTED_CONFIG_ORDER
    for seed in EXPECTED_SEEDS
)
EXCLUDED_RUN_IDS = ("F-RELU-s0-r2",)

RESULT_FIELDS = (
    "evaluation_id",
    "source_run_id",
    "variable",
    "config_id",
    "activation",
    "beta",
    "depth",
    "seed",
    "checkpoint_path",
    "checkpoint_hash",
    "final_weights_hash",
    "source_commit",
    "val_accuracy",
    "parameters",
    "training_flops_total",
    "test_samples",
    "test_correct",
    "test_accuracy",
    "inference_flops_total",
    "inference_flops_per_sample",
    "predictions_hash",
    "status",
)


@dataclass(frozen=True)
class CheckpointSpec:
    """One primary checkpoint and its frozen experimental metadata."""

    family: str
    config: dict[str, Any]
    configuration: dict[str, Any]
    row: dict[str, str]
    checkpoint_path: Path
    log_path: Path

    @property
    def run_id(self) -> str:
        return self.row["run_id"]

    @property
    def config_id(self) -> str:
        return self.row["config_id"]

    @property
    def seed(self) -> int:
        return int(self.row["seed"])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_paths(artifacts_dir: Path) -> dict[str, Path]:
    return {
        "input": artifacts_dir / INPUT_MANIFEST_NAME,
        "log": artifacts_dir / LOG_NAME,
        "results": artifacts_dir / RESULTS_NAME,
        "output": artifacts_dir / OUTPUT_MANIFEST_NAME,
        "lock": artifacts_dir / LOCK_NAME,
    }


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        run_v1._fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(
            descriptor,
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
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
        run_v1._fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _append_log(handle: Any, event: dict[str, Any]) -> None:
    handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _read_result_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
            raise ValueError("official results header differs from the frozen schema")
        return list(reader)


def _critical_source_paths() -> list[Path]:
    paths = list((REPO_ROOT / "bert_cpu").glob("*.py"))
    paths.extend(
        [
            REPO_ROOT / "datasets/__init__.py",
            REPO_ROOT / "datasets/loaders.py",
            REPO_ROOT / "datasets/adult/adult.data",
            REPO_ROOT / "datasets/adult/adult.test",
            REPO_ROOT / "exercises/q01_activations.py",
            REPO_ROOT / "exercises/task_binary_classification.py",
            REPO_ROOT / "experiments/run_v1.py",
            REPO_ROOT / "experiments/run_v2.py",
            REPO_ROOT / "experiments/run_v3.py",
            REPO_ROOT / "experiments/plot_v1.py",
            REPO_ROOT / "experiments/plot_v2.py",
            REPO_ROOT / "experiments/plot_v3.py",
            REPO_ROOT / "experiments/configs/v1_activation_family.json",
            REPO_ROOT / "experiments/configs/v2_softplus_curvature.json",
            REPO_ROOT / "experiments/configs/v3_linear_depth.json",
            REPO_ROOT / "requirements.txt",
        ]
    )
    return [path for path in paths if path.exists()]


def _source_paths() -> list[Path]:
    paths = _critical_source_paths()
    paths.append(Path(__file__).resolve())
    test_path = REPO_ROOT / "test/test_evaluate_official_test.py"
    if test_path.exists():
        paths.append(test_path)
    return paths


def _relative_paths(paths: Sequence[Path]) -> list[str]:
    return [
        path.resolve().relative_to(REPO_ROOT).as_posix()
        for path in paths
    ]


def _source_status(paths: Sequence[Path]) -> list[str]:
    relative = _relative_paths(paths)
    if not relative:
        return []
    output = run_v1._git(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *relative,
    )
    return output.splitlines() if output else []


def _feature_schema_hash(dataset: Any) -> str:
    return run_v1._hash_json(
        {
            "feature_names": list(dataset.feature_names),
            "categories": {
                name: list(values)
                for name, values in dataset.categories.items()
            },
        }
    )


def _validate_path(
    recorded: str,
    expected: Path,
    *,
    label: str,
) -> Path:
    path = run_v1._recorded_path(recorded)
    if path.is_symlink() or path.resolve() != expected.resolve():
        raise ValueError(f"{label} path is not canonical: {recorded}")
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {recorded}")
    return path


def _is_ancestor(commit: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def _validate_primary_row(
    family: str,
    config: dict[str, Any],
    selected: dict[str, Any],
    row: dict[str, str],
    *,
    root: Path,
) -> CheckpointSpec:
    config_id = selected["id"]
    seed = int(row["seed"])
    run_id = f"{config_id}-s{seed}-r1"
    expected_activation = selected["activation"]
    expected_parameters = int(
        selected.get("parameters", config["architecture"].get("parameters", 0))
    )
    expected_initial = (
        config["data"]["initial_weights_sha256"][config_id][str(seed)]
        if family == "v3"
        else config["data"]["initial_weights_sha256"][str(seed)]
    )
    expected_phase = {
        "v1": "v1_train_validation",
        "v2": "v2_train_validation",
        "v3": "v3_train_validation",
    }[family]

    if (
        row["run_id"] != run_id
        or row["task"] != "adult_binary_classification"
        or row["variable"] != config["variable"]
        or row["config_id"] != config_id
        or row["activation"] != expected_activation
        or seed not in EXPECTED_SEEDS
        or int(row["repetition"]) != 1
        or row["run_kind"] != "scientific"
        or row["phase"] != expected_phase
        or row["purpose"] != "primary"
        or row["status"] != "completed_valid"
        or str(row["test_accuracy"]).strip()
        or int(row["epochs"]) != 100
        or row["optimizer"] != "Adam"
        or float(row["learning_rate"]) != 0.01
        or int(row["train_samples"]) != 26_049
        or int(row["val_samples"]) != EXPECTED_VALIDATION_SAMPLES
        or int(row["parameters"]) != expected_parameters
        or row["dataset_hash"] != config["data"]["encoded_train_sha256"]
        or int(row["split_seed"]) != 0
        or row["split_hash"] != config["data"]["split_sha256"]
        or row["initial_weights_hash"] != expected_initial
        or int(row["flops_per_epoch"])
        != int(selected["expected_flops_per_epoch"])
        or int(row["flops_total"])
        != int(selected["expected_flops_per_epoch"]) * 100
        or int(row["inference_flops_total"])
        != int(selected["expected_inference_flops_total"])
        or float(row["inference_flops_per_sample"])
        != int(selected["expected_inference_flops_total"])
        / EXPECTED_VALIDATION_SAMPLES
        or row["config_hash"]
        != run_v1.sha256_file(
            {
                "v1": run_v1.DEFAULT_CONFIG,
                "v2": run_v2.DEFAULT_CONFIG,
                "v3": run_v3.DEFAULT_CONFIG,
            }[family]
        )
        or not _is_ancestor(row["commit"])
    ):
        raise ValueError(f"primary row differs from frozen protocol: {run_id}")

    if family == "v2" and float(row["beta"]) != float(selected["beta"]):
        raise ValueError(f"beta differs from frozen protocol: {run_id}")
    if family == "v3" and (
        int(row["depth"]) != int(selected["depth"])
        or json.loads(row["layer_sizes"]) != selected["layer_sizes"]
    ):
        raise ValueError(f"depth differs from frozen protocol: {run_id}")

    checkpoint_path = _validate_path(
        row["checkpoint_path"],
        root / "checkpoints" / f"{run_id}.npz",
        label="checkpoint",
    )
    log_path = _validate_path(
        row["log_path"],
        root / "logs" / f"{run_id}.jsonl",
        label="log",
    )
    return CheckpointSpec(
        family=family,
        config=config,
        configuration=selected,
        row=row,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
    )


def collect_checkpoint_specs() -> tuple[CheckpointSpec, ...]:
    """Validate and return the exact 33-checkpoint primary grid."""
    # These loaders perform their own complete log/checkpoint validation.
    if sum(map(len, plot_v1.load_v1_runs().values())) != 12:
        raise ValueError("V1 strict loader did not return 12 primary runs")
    if sum(map(len, plot_v2.load_v2_runs().values())) != 12:
        raise ValueError("V2 strict loader did not return 12 primary runs")
    if sum(map(len, plot_v3.load_v3_runs().values())) != 9:
        raise ValueError("V3 strict loader did not return 9 primary runs")

    v1_config = run_v1.load_config()
    v2_config = run_v2.load_config()
    v3_config = run_v3.load_config()
    run_v1.validate_frozen_config(v1_config)
    run_v2.validate_frozen_config(v2_config)
    run_v3.validate_frozen_config(v3_config)

    v1_rows = run_v1._read_results(run_v1.DEFAULT_ARTIFACTS / "results.csv")
    v2_rows = run_v2.load_results()
    v3_rows = run_v3.load_results()
    expected_v1_all = set(EXPECTED_RUN_IDS[:12]) | set(EXCLUDED_RUN_IDS)
    if (
        len(v1_rows) != 13
        or {row["run_id"] for row in v1_rows} != expected_v1_all
    ):
        raise ValueError("V1 contains an unexpected, missing or duplicated run")
    if (
        len(v2_rows) != 12
        or {row["run_id"] for row in v2_rows}
        != set(EXPECTED_RUN_IDS[12:24])
    ):
        raise ValueError("V2 contains an unexpected, missing or duplicated run")
    if (
        len(v3_rows) != 9
        or {row["run_id"] for row in v3_rows}
        != set(EXPECTED_RUN_IDS[24:])
    ):
        raise ValueError("V3 contains an unexpected, missing or duplicated run")

    repeat = next(row for row in v1_rows if row["run_id"] == EXCLUDED_RUN_IDS[0])
    if (
        repeat["purpose"] != "determinism_repeat"
        or not run_v1._result_artifacts_valid(repeat)
    ):
        raise ValueError("the excluded ReLU repetition is invalid")

    bundles = (
        (
            "v1",
            v1_config,
            v1_rows,
            run_v1.DEFAULT_ARTIFACTS,
        ),
        (
            "v2",
            v2_config,
            v2_rows,
            run_v2.DEFAULT_ARTIFACTS,
        ),
        (
            "v3",
            v3_config,
            v3_rows,
            run_v3.DEFAULT_ARTIFACTS,
        ),
    )
    validators: dict[str, Callable[[dict[str, Any]], bool]] = {
        "v1": run_v1._result_artifacts_valid,
        "v2": run_v2.result_artifacts_valid,
        "v3": run_v3.result_artifacts_valid,
    }
    specs: list[CheckpointSpec] = []
    for family, config, rows, root in bundles:
        row_by_id = {row["run_id"]: row for row in rows}
        for selected in config["configurations"]:
            for seed in EXPECTED_SEEDS:
                run_id = f"{selected['id']}-s{seed}-r1"
                row = row_by_id[run_id]
                if not validators[family](row):
                    raise ValueError(f"source artifact validator rejected {run_id}")
                specs.append(
                    _validate_primary_row(
                        family,
                        config,
                        selected,
                        row,
                        root=Path(root),
                    )
                )

    observed_ids = tuple(spec.run_id for spec in specs)
    if observed_ids != EXPECTED_RUN_IDS:
        raise ValueError("checkpoint order differs from the frozen official grid")
    if len({spec.run_id for spec in specs}) != 33:
        raise ValueError("official grid contains duplicated run IDs")
    return tuple(specs)


def _build_model(spec: CheckpointSpec) -> Any:
    if spec.family == "v3":
        model = adult.AdultLinearClassifier(
            EXPECTED_TEST_FEATURES,
            depth=int(spec.configuration["depth"]),
            hidden=64,
        )
    else:
        beta = (
            float(spec.configuration["beta"])
            if spec.family == "v2"
            else None
        )
        model = adult.AdultMLP(
            EXPECTED_TEST_FEATURES,
            hidden=64,
            activation=spec.configuration["activation"],
            activation_beta=beta,
        )
    return model.eval()


def _load_checkpoint_strict(spec: CheckpointSpec, model: Any) -> dict[str, Any]:
    if run_v1.sha256_file(spec.checkpoint_path) != spec.row["checkpoint_hash"]:
        raise ValueError(f"checkpoint SHA-256 changed: {spec.run_id}")

    named = list(model.named_parameters())
    expected_names = [name for name, _ in named]
    expected_shapes = {
        name: list(parameter.shape)
        for name, parameter in named
    }
    with np.load(spec.checkpoint_path, allow_pickle=False) as archive:
        expected_archive_keys = {
            "metadata_json",
            *(f"parameter_{index}" for index in range(len(named))),
        }
        if set(archive.files) != expected_archive_keys:
            raise ValueError(f"checkpoint keys changed: {spec.run_id}")
        manifest = json.loads(str(archive["metadata_json"].item()))
        if manifest.get("parameter_names") != expected_names:
            raise ValueError(f"checkpoint parameter names changed: {spec.run_id}")
        if manifest.get("parameter_shapes") != expected_shapes:
            raise ValueError(f"checkpoint parameter shapes changed: {spec.run_id}")

        common_expected = {
            "run_id": spec.run_id,
            "run_kind": "scientific",
            "config_id": spec.config_id,
            "activation": spec.row["activation"],
            "seed": spec.seed,
            "repetition": 1,
            "purpose": "primary",
            "split_hash": spec.row["split_hash"],
            "config_hash": spec.row["config_hash"],
            "environment_hash": spec.row["environment_hash"],
            "commit": spec.row["commit"],
            "code_state_hash": spec.row["code_state_hash"],
            "initial_weights_hash": spec.row["initial_weights_hash"],
            "final_weights_hash": spec.row["final_weights_hash"],
            "epoch": 100,
            "configuration": spec.configuration,
            "training": spec.config["training"],
        }
        if any(manifest.get(key) != value for key, value in common_expected.items()):
            raise ValueError(f"checkpoint manifest changed: {spec.run_id}")
        if spec.family == "v2" and (
            float(manifest.get("beta")) != float(spec.configuration["beta"])
            or manifest.get("variable") != spec.config["variable"]
            or manifest.get("phase") != spec.config["phase"]
        ):
            raise ValueError(f"V2 checkpoint manifest changed: {spec.run_id}")
        if spec.family == "v3" and (
            manifest.get("depth") != spec.configuration["depth"]
            or manifest.get("layer_sizes") != spec.configuration["layer_sizes"]
            or manifest.get("variable") != spec.config["variable"]
            or manifest.get("phase") != spec.config["phase"]
        ):
            raise ValueError(f"V3 checkpoint manifest changed: {spec.run_id}")

        stored: list[np.ndarray] = []
        for index, (_, parameter) in enumerate(named):
            array = archive[f"parameter_{index}"].copy()
            if (
                array.shape != parameter.shape
                or array.dtype != parameter.data.dtype
                or array.dtype != np.dtype(np.float64)
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"checkpoint parameter invalid: {spec.run_id}")
            stored.append(array)

    weights_hash = run_v1.hash_arrays(*stored)
    if (
        weights_hash != spec.row["final_weights_hash"]
        or weights_hash != manifest["final_weights_hash"]
    ):
        raise ValueError(f"checkpoint weights hash changed: {spec.run_id}")
    for array, (_, parameter) in zip(stored, named):
        parameter.data[...] = array
    if run_v1.parameter_hash(model) != weights_hash:
        raise ValueError(f"checkpoint restore failed: {spec.run_id}")
    return manifest


def _forward_once(
    spec: CheckpointSpec,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    model = _build_model(spec)
    _load_checkpoint_strict(spec, model)
    weights_before = run_v1.parameter_hash(model)

    cpu.reset_flops()
    logits = model(cpu.Tensor(X, requires_grad=False)).data
    measured_flops = cpu.flop_count()

    if logits.shape != (2, X.shape[1]) or not np.isfinite(logits).all():
        raise FloatingPointError(f"invalid logits for {spec.run_id}")
    predictions = logits.argmax(axis=0).astype(np.int64, copy=False)
    correct = int((predictions == y).sum())
    if run_v1.parameter_hash(model) != weights_before:
        raise RuntimeError(f"inference mutated weights for {spec.run_id}")
    return {
        "correct": correct,
        "accuracy": correct / int(y.size),
        "flops": int(measured_flops),
        "predictions_hash": run_v1.hash_arrays(predictions),
    }


def _checkpoint_manifest_record(
    spec: CheckpointSpec,
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_run_id": spec.run_id,
        "family": spec.family,
        "variable": spec.row["variable"],
        "config_id": spec.config_id,
        "activation": spec.row["activation"],
        "beta": (
            float(spec.configuration["beta"])
            if spec.family == "v2"
            else None
        ),
        "depth": (
            int(spec.configuration["depth"])
            if spec.family == "v3"
            else None
        ),
        "seed": spec.seed,
        "checkpoint_path": _display_path(spec.checkpoint_path),
        "checkpoint_hash": spec.row["checkpoint_hash"],
        "final_weights_hash": spec.row["final_weights_hash"],
        "log_path": _display_path(spec.log_path),
        "log_hash": run_v1.sha256_file(spec.log_path),
        "source_commit": spec.row["commit"],
        "source_code_state_hash": spec.row["code_state_hash"],
        "source_config_hash": spec.row["config_hash"],
        "source_environment_hash": spec.row["environment_hash"],
        "source_split_hash": spec.row["split_hash"],
        "source_val_accuracy": float(spec.row["val_accuracy"]),
        "source_parameters": int(spec.row["parameters"]),
        "source_training_flops_total": int(spec.row["flops_total"]),
        "validation_reproduced_accuracy": validation["accuracy"],
        "validation_reproduced_flops": validation["flops"],
        "validation_predictions_hash": validation["predictions_hash"],
    }


def build_input_manifest() -> tuple[dict[str, Any], tuple[CheckpointSpec, ...]]:
    """Run the full no-test preflight and return its deterministic manifest."""
    critical_paths = _critical_source_paths()
    critical_status = _source_status(critical_paths)
    if critical_status:
        raise RuntimeError(
            "inference-critical versioned sources are dirty: "
            + "; ".join(critical_status)
        )

    raw_train_path = REPO_ROOT / "datasets/adult/adult.data"
    raw_test_path = REPO_ROOT / "datasets/adult/adult.test"
    raw_train_hash = run_v1.sha256_file(raw_train_path)
    raw_test_hash = run_v1.sha256_file(raw_test_path)
    raw_test_blob = run_v1._git("hash-object", _display_path(raw_test_path))
    committed_test_blob = run_v1._git(
        "rev-parse",
        f"HEAD:{_display_path(raw_test_path)}",
    )
    if raw_train_hash != EXPECTED_RAW_TRAIN_SHA256:
        raise ValueError("raw Adult training file changed")
    if (
        raw_test_hash != EXPECTED_RAW_TEST_SHA256
        or raw_test_blob != EXPECTED_RAW_TEST_GIT_BLOB
        or committed_test_blob != EXPECTED_RAW_TEST_GIT_BLOB
    ):
        raise ValueError("official Adult test file identity changed")

    specs = collect_checkpoint_specs()
    train_dataset = datasets.load_adult("train")
    if (
        train_dataset.X.shape != (108, 32_561)
        or train_dataset.y.shape != (32_561,)
        or not np.isfinite(train_dataset.X).all()
        or set(np.unique(train_dataset.y).tolist()) != {0, 1}
    ):
        raise ValueError("encoded Adult training data changed")
    encoded_train_hash = run_v1.hash_arrays(train_dataset.X, train_dataset.y)
    if encoded_train_hash != specs[0].config["data"]["encoded_train_sha256"]:
        raise ValueError("encoded Adult training hash changed")

    _, _, X_val, y_val = adult.train_val_split(
        train_dataset.X,
        train_dataset.y,
        val_frac=0.2,
        split_seed=0,
    )
    records: list[dict[str, Any]] = []
    validation_predictions: dict[tuple[str, int], str] = {}
    for spec in specs:
        validation = _forward_once(spec, X_val, y_val)
        if (
            validation["accuracy"] != float(spec.row["val_accuracy"])
            or validation["flops"] != int(spec.row["inference_flops_total"])
        ):
            raise ValueError(
                f"validation checkpoint reproduction failed: {spec.run_id}"
            )
        records.append(_checkpoint_manifest_record(spec, validation))
        validation_predictions[(spec.config_id, spec.seed)] = validation[
            "predictions_hash"
        ]

    for seed in EXPECTED_SEEDS:
        if (
            validation_predictions[("F-SOFTPLUS", seed)]
            != validation_predictions[("S-BETA-1", seed)]
        ):
            raise ValueError("Softplus and Softplus-beta=1 validation predictions differ")

    source_paths = _source_paths()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "phase": PHASE,
        "status": "preflight_valid",
        "policy": {
            "all_primary_checkpoints_required": True,
            "configuration_selection_from_test_forbidden": True,
            "training_forbidden": True,
            "single_test_load": True,
            "single_forward_per_checkpoint": True,
            "source_results_immutable": True,
        },
        "branch": run_v1._git("branch", "--show-current"),
        "evaluation_commit": run_v1._git("rev-parse", "HEAD"),
        "environment": run_v1._environment_metadata(),
        "source_state_hash": run_v1._hash_files(source_paths),
        "source_status": _source_status(source_paths),
        "critical_source_status": critical_status,
        "evaluator_hash": run_v1.sha256_file(Path(__file__).resolve()),
        "test_module_hash": (
            run_v1.sha256_file(
                REPO_ROOT / "test/test_evaluate_official_test.py"
            )
            if (REPO_ROOT / "test/test_evaluate_official_test.py").exists()
            else None
        ),
        "source_artifacts": {
            "v1_config_hash": run_v1.sha256_file(run_v1.DEFAULT_CONFIG),
            "v1_results_hash": run_v1.sha256_file(
                run_v1.DEFAULT_ARTIFACTS / "results.csv"
            ),
            "v2_config_hash": run_v1.sha256_file(run_v2.DEFAULT_CONFIG),
            "v2_results_hash": run_v1.sha256_file(
                run_v2.DEFAULT_ARTIFACTS / "results.csv"
            ),
            "v3_config_hash": run_v1.sha256_file(run_v3.DEFAULT_CONFIG),
            "v3_results_hash": run_v1.sha256_file(
                run_v3.DEFAULT_ARTIFACTS / "results.csv"
            ),
        },
        "training_data": {
            "raw_hash": raw_train_hash,
            "encoded_hash": encoded_train_hash,
            "features": int(train_dataset.n_features),
            "samples": int(train_dataset.n_samples),
            "validation_samples": int(y_val.size),
            "feature_schema_hash": _feature_schema_hash(train_dataset),
            "split_hash": specs[0].row["split_hash"],
        },
        "official_test_identity": {
            "raw_path": _display_path(raw_test_path),
            "expected_raw_sha256": raw_test_hash,
            "expected_git_blob": raw_test_blob,
            "content_parsed": False,
            "labels_or_metrics_consulted": False,
        },
        "expected_config_order": list(EXPECTED_CONFIG_ORDER),
        "expected_run_ids": list(EXPECTED_RUN_IDS),
        "excluded_run_ids": list(EXCLUDED_RUN_IDS),
        "checkpoint_count": len(records),
        "checkpoints": records,
    }
    evaluation_id = f"OFFICIAL-{run_v1._hash_json(payload)[:20]}"
    manifest = {"evaluation_id": evaluation_id, **payload}
    return manifest, specs


def prepare_input_manifest(
    artifacts_dir: Path = DEFAULT_ARTIFACTS,
) -> dict[str, Any]:
    """Persist or revalidate the deterministic preflight manifest."""
    paths = _artifact_paths(artifacts_dir)
    manifest, _ = build_input_manifest()
    if paths["input"].exists():
        recorded = _read_json(paths["input"])
        if recorded != manifest:
            raise RuntimeError("existing input manifest differs from current preflight")
        return recorded
    if any(paths[name].exists() for name in ("log", "results", "output")):
        raise RuntimeError("partial official-evaluation artifacts already exist")
    _atomic_write_json(paths["input"], manifest)
    return manifest


def _official_test_metadata(test_dataset: Any) -> dict[str, Any]:
    if (
        test_dataset.split != "test"
        or test_dataset.X.shape
        != (EXPECTED_TEST_FEATURES, EXPECTED_TEST_SAMPLES)
        or test_dataset.y.shape != (EXPECTED_TEST_SAMPLES,)
        or test_dataset.X.dtype != np.dtype(np.float64)
        or test_dataset.y.dtype != np.dtype(np.int64)
        or not np.isfinite(test_dataset.X).all()
        or set(np.unique(test_dataset.y).tolist()) != {0, 1}
    ):
        raise ValueError("official Adult test data has an unexpected schema")
    raw_path = REPO_ROOT / "datasets/adult/adult.test"
    raw_hash = run_v1.sha256_file(raw_path)
    raw_blob = run_v1._git("hash-object", _display_path(raw_path))
    if (
        raw_hash != EXPECTED_RAW_TEST_SHA256
        or raw_blob != EXPECTED_RAW_TEST_GIT_BLOB
    ):
        raise ValueError("loaded Adult test does not match the frozen identity")
    return {
        "raw_path": _display_path(raw_path),
        "raw_sha256": raw_hash,
        "git_blob": raw_blob,
        "encoded_hash": run_v1.hash_arrays(test_dataset.X, test_dataset.y),
        "features_hash": run_v1.hash_arrays(test_dataset.X),
        "labels_hash": run_v1.hash_arrays(test_dataset.y),
        "feature_schema_hash": _feature_schema_hash(test_dataset),
        "features": int(test_dataset.n_features),
        "samples": int(test_dataset.n_samples),
        "positive_fraction": float(test_dataset.y.mean()),
        "majority_accuracy": float(
            max(test_dataset.y.mean(), 1.0 - test_dataset.y.mean())
        ),
    }


def _official_result_row(
    evaluation_id: str,
    spec: CheckpointSpec,
    result: dict[str, Any],
    test_samples: int,
) -> dict[str, Any]:
    per_sample = int(spec.row["inference_flops_total"]) / int(
        spec.row["val_samples"]
    )
    expected_flops = int(per_sample * test_samples)
    if per_sample != int(per_sample) or result["flops"] != expected_flops:
        raise ValueError(f"official inference FLOPs changed: {spec.run_id}")
    return {
        "evaluation_id": evaluation_id,
        "source_run_id": spec.run_id,
        "variable": spec.row["variable"],
        "config_id": spec.config_id,
        "activation": spec.row["activation"],
        "beta": (
            float(spec.configuration["beta"])
            if spec.family == "v2"
            else ""
        ),
        "depth": (
            int(spec.configuration["depth"])
            if spec.family == "v3"
            else ""
        ),
        "seed": spec.seed,
        "checkpoint_path": _display_path(spec.checkpoint_path),
        "checkpoint_hash": spec.row["checkpoint_hash"],
        "final_weights_hash": spec.row["final_weights_hash"],
        "source_commit": spec.row["commit"],
        "val_accuracy": float(spec.row["val_accuracy"]),
        "parameters": int(spec.row["parameters"]),
        "training_flops_total": int(spec.row["flops_total"]),
        "test_samples": test_samples,
        "test_correct": result["correct"],
        "test_accuracy": result["accuracy"],
        "inference_flops_total": result["flops"],
        "inference_flops_per_sample": int(per_sample),
        "predictions_hash": result["predictions_hash"],
        "status": "completed_valid",
    }


def _validate_result_rows(
    rows: Sequence[dict[str, Any] | dict[str, str]],
    evaluation_id: str,
) -> None:
    if len(rows) != 33:
        raise ValueError("official evaluation must contain exactly 33 rows")
    if tuple(row["source_run_id"] for row in rows) != EXPECTED_RUN_IDS:
        raise ValueError("official result order or identity changed")
    for row in rows:
        if (
            row["evaluation_id"] != evaluation_id
            or row["status"] != "completed_valid"
            or int(row["test_samples"]) != EXPECTED_TEST_SAMPLES
            or not 0.0 <= float(row["test_accuracy"]) <= 1.0
            or int(row["test_correct"]) / int(row["test_samples"])
            != float(row["test_accuracy"])
            or int(row["inference_flops_total"])
            != int(row["inference_flops_per_sample"])
            * int(row["test_samples"])
        ):
            raise ValueError(f"invalid official result row: {row['source_run_id']}")


def _typed_result_row(row: dict[str, Any] | dict[str, str]) -> dict[str, Any]:
    typed = dict(row)
    for field in (
        "seed",
        "parameters",
        "training_flops_total",
        "test_samples",
        "test_correct",
        "inference_flops_total",
        "inference_flops_per_sample",
    ):
        typed[field] = int(typed[field])
    for field in ("val_accuracy", "test_accuracy"):
        typed[field] = float(typed[field])
    if typed["beta"] not in ("", None):
        typed["beta"] = float(typed["beta"])
    else:
        typed["beta"] = ""
    if typed["depth"] not in ("", None):
        typed["depth"] = int(typed["depth"])
    else:
        typed["depth"] = ""
    return typed


def _validate_semantic_links(
    input_manifest: dict[str, Any],
    output_manifest: dict[str, Any],
    rows: Sequence[dict[str, str]],
    events: Sequence[dict[str, Any]],
) -> None:
    source_records = input_manifest.get("checkpoints", [])
    if (
        len(source_records) != 33
        or tuple(item.get("source_run_id") for item in source_records)
        != EXPECTED_RUN_IDS
    ):
        raise ValueError("input manifest checkpoint inventory is invalid")

    test_events = [
        event for event in events if event.get("event") == "official_test_loaded"
    ]
    result_events = [
        event for event in events if event.get("event") == "checkpoint_evaluated"
    ]
    if (
        len(test_events) != 1
        or test_events[0].get("data") != output_manifest.get("test_data")
        or len(result_events) != 33
    ):
        raise ValueError("official log is not linked to the output manifest")

    typed_rows = [_typed_result_row(row) for row in rows]
    for source, row, event in zip(source_records, typed_rows, result_events):
        expected_beta = source["beta"] if source["beta"] is not None else ""
        expected_depth = source["depth"] if source["depth"] is not None else ""
        expected_per_sample = int(
            source["validation_reproduced_flops"]
            / EXPECTED_VALIDATION_SAMPLES
        )
        expected = {
            "source_run_id": source["source_run_id"],
            "variable": source["variable"],
            "config_id": source["config_id"],
            "activation": source["activation"],
            "beta": expected_beta,
            "depth": expected_depth,
            "seed": source["seed"],
            "checkpoint_path": source["checkpoint_path"],
            "checkpoint_hash": source["checkpoint_hash"],
            "final_weights_hash": source["final_weights_hash"],
            "source_commit": source["source_commit"],
            "val_accuracy": source["source_val_accuracy"],
            "parameters": source["source_parameters"],
            "training_flops_total": source["source_training_flops_total"],
            "inference_flops_per_sample": expected_per_sample,
            "inference_flops_total": expected_per_sample * EXPECTED_TEST_SAMPLES,
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise ValueError(
                f"official row differs from input manifest: {row['source_run_id']}"
            )
        typed_event = _typed_result_row(
            {field: event.get(field, "") for field in RESULT_FIELDS}
        )
        if typed_event != row:
            raise ValueError(
                f"official log differs from CSV: {row['source_run_id']}"
            )

    completed = [
        event for event in events if event.get("event") == "evaluation_completed"
    ]
    if (
        len(completed) != 1
        or completed[0].get("status") != "completed_valid"
        or completed[0].get("checkpoint_count") != 33
        or completed[0].get("results_sha256")
        != output_manifest.get("results_sha256")
    ):
        raise ValueError("completion event differs from final output")


def verify_saved_artifacts(
    artifacts_dir: Path = DEFAULT_ARTIFACTS,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate final artifacts without loading Adult test."""
    paths = _artifact_paths(artifacts_dir)
    if not all(paths[name].is_file() for name in ("input", "log", "results", "output")):
        raise FileNotFoundError("official evaluation is incomplete")
    output = _read_json(paths["output"])
    input_manifest = _read_json(paths["input"])
    evaluation_id = input_manifest["evaluation_id"]
    if (
        output.get("schema_version") != 1
        or output.get("phase") != PHASE
        or output.get("status") != "completed_valid"
        or output.get("evaluation_id") != evaluation_id
        or int(output.get("checkpoint_count", -1)) != 33
        or output.get("official_test_loaded_count") != 1
        or output.get("input_manifest_sha256")
        != run_v1.sha256_file(paths["input"])
        or output.get("evaluation_log_sha256")
        != run_v1.sha256_file(paths["log"])
        or output.get("results_sha256")
        != run_v1.sha256_file(paths["results"])
    ):
        raise ValueError("official output manifest is invalid")

    rows = _read_result_csv(paths["results"])
    _validate_result_rows(rows, evaluation_id)
    events = [
        json.loads(line)
        for line in paths["log"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        sum(event.get("event") == "official_test_loaded" for event in events) != 1
        or sum(event.get("event") == "checkpoint_evaluated" for event in events)
        != 33
        or not events
        or events[-1].get("event") != "evaluation_completed"
    ):
        raise ValueError("official evaluation log is incomplete")
    _validate_semantic_links(input_manifest, output, rows, events)
    return output, rows


def run_official_evaluation(
    artifacts_dir: Path = DEFAULT_ARTIFACTS,
    *,
    test_loader: Callable[[str], Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Evaluate all checkpoints, or return an already valid complete result."""
    paths = _artifact_paths(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with paths["lock"].open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if paths["output"].exists():
            return verify_saved_artifacts(artifacts_dir)
        if any(paths[name].exists() for name in ("log", "results")):
            raise RuntimeError(
                "partial official-evaluation artifacts exist; refusing to overwrite"
            )
        if not paths["input"].is_file():
            raise RuntimeError("run --preflight before loading the official test")

        current_manifest, specs = build_input_manifest()
        recorded_manifest = _read_json(paths["input"])
        if current_manifest != recorded_manifest:
            raise RuntimeError("preflight changed after the input manifest was frozen")
        evaluation_id = recorded_manifest["evaluation_id"]
        load_dataset = test_loader or datasets.load_adult

        with paths["log"].open("x", encoding="utf-8") as log:
            run_v1._fsync_directory(paths["log"].parent)
            _append_log(
                log,
                {
                    "event": "preflight_verified",
                    "at_utc": _utc_now(),
                    "evaluation_id": evaluation_id,
                    "checkpoint_count": len(specs),
                    "training_performed": False,
                },
            )
            try:
                test_dataset = load_dataset("test")
                test_metadata = _official_test_metadata(test_dataset)
                if (
                    test_metadata["feature_schema_hash"]
                    != recorded_manifest["training_data"]["feature_schema_hash"]
                ):
                    raise ValueError("official test feature schema differs from train")
                _append_log(
                    log,
                    {
                        "event": "official_test_loaded",
                        "at_utc": _utc_now(),
                        "evaluation_id": evaluation_id,
                        "data": test_metadata,
                    },
                )

                rows: list[dict[str, Any]] = []
                prediction_hashes: dict[tuple[str, int], str] = {}
                for spec in specs:
                    result = _forward_once(
                        spec,
                        test_dataset.X,
                        test_dataset.y,
                    )
                    row = _official_result_row(
                        evaluation_id,
                        spec,
                        result,
                        test_dataset.n_samples,
                    )
                    rows.append(row)
                    prediction_hashes[(spec.config_id, spec.seed)] = result[
                        "predictions_hash"
                    ]
                    _append_log(
                        log,
                        {
                            "event": "checkpoint_evaluated",
                            "at_utc": _utc_now(),
                            **row,
                        },
                    )

                for seed in EXPECTED_SEEDS:
                    if (
                        prediction_hashes[("F-SOFTPLUS", seed)]
                        != prediction_hashes[("S-BETA-1", seed)]
                    ):
                        raise ValueError(
                            "Softplus and Softplus-beta=1 test predictions differ"
                        )
                _validate_result_rows(rows, evaluation_id)
                post_manifest, _ = build_input_manifest()
                if post_manifest != recorded_manifest:
                    raise RuntimeError(
                        "source artifacts changed during official evaluation"
                    )
                _atomic_write_csv(paths["results"], rows)
                results_hash = run_v1.sha256_file(paths["results"])
                _append_log(
                    log,
                    {
                        "event": "evaluation_completed",
                        "at_utc": _utc_now(),
                        "evaluation_id": evaluation_id,
                        "checkpoint_count": len(rows),
                        "results_sha256": results_hash,
                        "status": "completed_valid",
                        "training_performed": False,
                    },
                )
            except Exception as error:
                _append_log(
                    log,
                    {
                        "event": "evaluation_failed",
                        "at_utc": _utc_now(),
                        "evaluation_id": evaluation_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise

        output_manifest = {
            "schema_version": 1,
            "phase": PHASE,
            "status": "completed_valid",
            "evaluation_id": evaluation_id,
            "completed_at_utc": _utc_now(),
            "checkpoint_count": 33,
            "configuration_count": 11,
            "seed_count": 3,
            "official_test_loaded_count": 1,
            "training_performed": False,
            "test_data": test_metadata,
            "input_manifest_path": _display_path(paths["input"]),
            "input_manifest_sha256": run_v1.sha256_file(paths["input"]),
            "evaluation_log_path": _display_path(paths["log"]),
            "evaluation_log_sha256": run_v1.sha256_file(paths["log"]),
            "results_path": _display_path(paths["results"]),
            "results_sha256": run_v1.sha256_file(paths["results"]),
            "environment": run_v1._environment_metadata(),
            "evaluation_commit": run_v1._git("rev-parse", "HEAD"),
            "source_state_hash": recorded_manifest["source_state_hash"],
        }
        _atomic_write_json(paths["output"], output_manifest)
        return verify_saved_artifacts(artifacts_dir)


def summarize_results(rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    """Return a descriptive per-configuration summary without selecting models."""
    summary: list[dict[str, Any]] = []
    for config_id in EXPECTED_CONFIG_ORDER:
        selected = [row for row in rows if row["config_id"] == config_id]
        values = np.asarray(
            [float(row["test_accuracy"]) * 100 for row in selected],
            dtype=float,
        )
        summary.append(
            {
                "config_id": config_id,
                "mean_test_accuracy_pct": float(values.mean()),
                "std_test_accuracy_pp": float(values.std(ddof=1)),
            }
        )
    return summary


def _print_summary(
    output: dict[str, Any],
    rows: Sequence[dict[str, str]],
) -> None:
    print(
        f"Official evaluation {output['evaluation_id']}: "
        f"{len(rows)} checkpoints, test loaded once, no training"
    )
    for item in summarize_results(rows):
        print(
            f"{item['config_id']:16s} "
            f"{item['mean_test_accuracy_pct']:.4f}% "
            f"+/- {item['std_test_accuracy_pp']:.4f} p.p."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight and run the all-checkpoint official Adult test.",
    )
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument(
        "--preflight",
        action="store_true",
        help="validate all 33 checkpoints without loading Adult test",
    )
    commands.add_argument(
        "--evaluate-official-test",
        action="store_true",
        help="load Adult test once and evaluate all 33 frozen checkpoints",
    )
    commands.add_argument(
        "--verify-only",
        action="store_true",
        help="validate saved final artifacts without loading Adult test",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Any:
    args = _build_parser().parse_args(argv)
    if args.preflight:
        manifest = prepare_input_manifest()
        print(
            f"Preflight {manifest['evaluation_id']}: "
            f"{manifest['checkpoint_count']} checkpoints valid; "
            "official test not loaded"
        )
        return manifest
    if args.evaluate_official_test:
        output, rows = run_official_evaluation()
        _print_summary(output, rows)
        return output, rows
    output, rows = verify_saved_artifacts()
    _print_summary(output, rows)
    return output, rows


if __name__ == "__main__":
    main()
