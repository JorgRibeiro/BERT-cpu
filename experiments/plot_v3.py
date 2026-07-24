"""Regenerate Variable 3 tables and plots from validated raw artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments import plot_v1, run_v3


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = Path(run_v3.DEFAULT_ARTIFACTS)
DEFAULT_RESULTS = DEFAULT_ARTIFACTS / "results.csv"
DEFAULT_V1_RESULTS = REPO_ROOT / "experiments/results.csv"
DEFAULT_SUMMARY = DEFAULT_ARTIFACTS / "summary.csv"
DEFAULT_PLOTS = DEFAULT_ARTIFACTS / "plots"

CONFIG_ORDER = run_v3.CONFIG_ORDER
LABELS = {
    "L1-DIRECT": "1 camada",
    "L2-IDENTITY": "2 camadas",
    "L3-IDENTITY": "3 camadas",
}
COLORS = {
    "L1-DIRECT": "#1f77b4",
    "L2-IDENTITY": "#ff7f0e",
    "L3-IDENTITY": "#d62728",
}
DIAGNOSTIC_EPOCHS = (0, 1, 25, 50, 75, 100)
RELEVANT_DIFFERENCE_PP = 0.5
REQUIRED_SEED_AGREEMENT = 2

CONTEXT_FIELDS = (
    "task",
    "variable",
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
    "epochs",
    "optimizer",
    "learning_rate",
    "train_samples",
    "val_samples",
)

SUMMARY_FIELDS = (
    "accuracy_rank",
    "config_id",
    "depth",
    "parameters",
    "mean_train_loss",
    "mean_val_loss",
    "mean_train_accuracy_pct",
    "mean_val_accuracy_pct",
    "std_val_accuracy_pp",
    "best_val_accuracy_seed0_pct",
    "best_epoch_seed0",
    "best_val_accuracy_seed1_pct",
    "best_epoch_seed1",
    "best_val_accuracy_seed2_pct",
    "best_epoch_seed2",
    "mean_best_val_accuracy_pct",
    "gflops_per_run",
    "return_pp_per_gflop",
    "pareto_status",
    "h3a_status",
    "h3a_l2_minus_l1_mean_pp",
    "h3a_l2_positive_seeds",
    "h3a_l2_negative_seeds",
    "h3a_l3_minus_l1_mean_pp",
    "h3a_l3_positive_seeds",
    "h3a_l3_negative_seeds",
    "h3b_status",
    "h3c_status",
    "h3d_status",
    "h3d_relu_minus_l2_mean_pp",
    "h3d_positive_seeds",
    "h3d_negative_seeds",
    "relu_bridge_same_platform",
)


@dataclass(frozen=True)
class RecordedRun:
    """One strictly validated V3 scientific run."""

    row: dict[str, str]
    epochs: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]
    majority_accuracy: float


@dataclass(frozen=True)
class H3aComparison:
    """One deeper-model minus direct-model paired comparison."""

    config_id: str
    mean_gain_pp: float
    positive_seeds: int
    negative_seeds: int
    status: str


@dataclass(frozen=True)
class H3Evaluation:
    """All four pre-experimental Variable 3 decisions."""

    h3a_comparisons: tuple[H3aComparison, ...]
    h3a_status: str
    h3b_status: str
    h3c_status: str
    h3d_status: str
    relu_minus_l2_mean_pp: float
    relu_minus_l2_positive_seeds: int
    relu_minus_l2_negative_seeds: int
    relu_bridge_same_platform: bool | None


def _path_from_row(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_recorded_run(row: dict[str, str]) -> RecordedRun:
    if not run_v3.result_artifacts_valid(row):
        raise ValueError(f"invalid artifacts for {row['run_id']}")
    if row["test_accuracy"].strip():
        raise ValueError("V3 analysis must not consume official-test accuracy")

    events = _read_events(_path_from_row(row["log_path"]))
    started = [event for event in events if event.get("event") == "run_started"]
    epochs = tuple(event for event in events if event.get("event") == "epoch")
    diagnostics = tuple(
        event for event in events if event.get("event") == "diagnostic"
    )
    if len(started) != 1:
        raise ValueError(f"run_started is missing or duplicated in {row['run_id']}")
    if started[0].get("data", {}).get("official_test_loaded") is not False:
        raise ValueError(f"official-test policy violated in {row['run_id']}")
    if [event.get("epoch") for event in epochs] != list(range(1, 101)):
        raise ValueError(f"incomplete epoch history in {row['run_id']}")
    if [event.get("epoch") for event in diagnostics] != list(DIAGNOSTIC_EPOCHS):
        raise ValueError(f"incomplete diagnostics in {row['run_id']}")
    if not all(event.get("equivalent_affine") is True for event in diagnostics):
        raise ValueError(f"affine equivalence failed in {row['run_id']}")

    final = epochs[-1]
    for row_field, event_field in {
        "train_loss_final": "train_loss",
        "val_loss_final": "val_loss",
        "train_accuracy": "train_accuracy",
        "val_accuracy": "val_accuracy",
    }.items():
        if float(row[row_field]) != float(final[event_field]):
            raise ValueError(f"final metric mismatch in {row['run_id']}")

    return RecordedRun(
        row=row,
        epochs=epochs,
        diagnostics=diagnostics,
        majority_accuracy=float(
            started[0]["data"]["validation_majority_accuracy"]
        ),
    )


def load_v3_runs(
    results_path: Path = DEFAULT_RESULTS,
) -> dict[str, tuple[RecordedRun, ...]]:
    """Load the complete three-depth by three-seed V3 grid."""
    rows = run_v3.load_results(results_path)
    if len(rows) != 9:
        raise ValueError("V3 must contain exactly nine scientific rows")
    if any(row["variable"] != "V3_linear_depth_without_activation" for row in rows):
        raise ValueError("V3 results contain a row from another variable")
    if any(
        row["status"] != "completed_valid"
        or row["run_kind"] != "scientific"
        or row["purpose"] != "primary"
        or int(row["epochs"]) != 100
        for row in rows
    ):
        raise ValueError("all V3 rows must be valid 100-epoch primary runs")

    for field in CONTEXT_FIELDS:
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"V3 runs mix experimental contexts: {field}")

    grouped: dict[str, tuple[RecordedRun, ...]] = {}
    for config_id in CONFIG_ORDER:
        selected = sorted(
            (row for row in rows if row["config_id"] == config_id),
            key=lambda row: int(row["seed"]),
        )
        if [int(row["seed"]) for row in selected] != [0, 1, 2]:
            raise ValueError(f"{config_id} does not contain seeds 0, 1 and 2")
        expected = run_v3.get_configuration(run_v3.load_config(), config_id)
        for row in selected:
            seed = int(row["seed"])
            if row["run_id"] != run_v3.expected_run_id(config_id, seed):
                raise ValueError(f"unexpected run identity: {row['run_id']}")
            if int(row["depth"]) != int(expected["depth"]):
                raise ValueError(f"depth differs from {config_id}")
            if json.loads(row["layer_sizes"]) != expected["layer_sizes"]:
                raise ValueError(f"layer sizes differ from {config_id}")
        grouped[config_id] = tuple(_load_recorded_run(row) for row in selected)

    majority = {run.majority_accuracy for runs in grouped.values() for run in runs}
    if len(majority) != 1:
        raise ValueError("validation majority accuracy changed between V3 runs")
    return grouped


def load_relu_reference(
    v1_results_path: Path = DEFAULT_V1_RESULTS,
) -> tuple[plot_v1.RecordedRun, ...]:
    """Load the three primary F-RELU runs through the frozen V1 validator."""
    return plot_v1.load_v1_runs(v1_results_path)["F-RELU"]


def validate_relu_bridge(
    grouped: Mapping[str, Sequence[RecordedRun]],
    relu_runs: Sequence[plot_v1.RecordedRun],
) -> bool:
    """Verify comparable controls and report whether the platform also matches."""
    l2_runs = grouped["L2-IDENTITY"]
    if len(l2_runs) != 3 or len(relu_runs) != 3:
        raise ValueError("ReLU bridge requires three paired seeds")

    common_fields = (
        "dataset_hash",
        "split_seed",
        "split_hash",
        "epochs",
        "optimizer",
        "learning_rate",
        "train_samples",
        "val_samples",
        "parameters",
    )
    data_fields = (
        "raw_train_hash",
        "encoded_train_hash",
        "features",
        "samples",
        "train_samples",
        "validation_samples",
        "split_seed",
        "split_hash",
        "validation_majority_accuracy",
        "preprocessing_limitation",
        "official_test_loaded",
    )
    environment_fields = (
        "python",
        "numpy",
        "default_dtype",
        "requirements_hash",
    )
    same_platform = True
    for l2, relu in zip(l2_runs, relu_runs):
        if int(l2.row["seed"]) != int(relu.row["seed"]):
            raise ValueError("ReLU bridge seeds are not paired")
        if any(l2.row[field] != relu.row[field] for field in common_fields):
            raise ValueError("ReLU bridge control mismatch")
        if l2.row["initial_weights_hash"] != relu.row["initial_weights_hash"]:
            raise ValueError("ReLU and L2 initial weights differ")
        if (
            l2.row["activation"] != "none"
            or int(l2.row["depth"]) != 2
            or json.loads(l2.row["layer_sizes"]) != [108, 64, 2]
            or relu.row["config_id"] != "F-RELU"
            or relu.row["activation"] != "relu"
        ):
            raise ValueError("ReLU bridge architecture or activation mismatch")
        if l2.majority_accuracy != relu.majority_accuracy:
            raise ValueError("ReLU bridge majority baseline mismatch")

        l2_started = next(
            event
            for event in _read_events(_path_from_row(l2.row["log_path"]))
            if event.get("event") == "run_started"
        )
        relu_started = next(
            event
            for event in _read_events(_path_from_row(relu.row["log_path"]))
            if event.get("event") == "run_started"
        )
        if any(
            l2_started["data"].get(field) != relu_started["data"].get(field)
            for field in data_fields
        ):
            raise ValueError("ReLU bridge data or preprocessing mismatch")
        if l2_started["training"] != relu_started["training"]:
            raise ValueError("ReLU bridge training procedure mismatch")
        if any(
            l2_started["environment"].get(field)
            != relu_started["environment"].get(field)
            for field in environment_fields
        ):
            raise ValueError("ReLU bridge software environment mismatch")
        same_platform &= (
            l2_started["environment"].get("platform")
            == relu_started["environment"].get("platform")
        )
        l2_architecture = l2_started["model"]["architecture"]
        relu_architecture = relu_started["model"]["architecture"]
        for field in (
            "input_features",
            "hidden_features",
            "output_classes",
            "parameters",
        ):
            if l2_architecture.get(field) != relu_architecture.get(field):
                raise ValueError("ReLU bridge linear architecture mismatch")
        if l2_started["model"].get("identity_operation_created") is not False:
            raise ValueError("L2 bridge unexpectedly records an Identity operation")
        if (
            int(relu.row["flops_per_epoch"]) - int(l2.row["flops_per_epoch"])
            != 2_083_904
        ):
            raise ValueError("ReLU-minus-L2 epoch FLOPs do not equal ReLU cost")
        if (
            int(relu.row["inference_flops_total"])
            - int(l2.row["inference_flops_total"])
            != 416_768
        ):
            raise ValueError("ReLU-minus-L2 inference FLOPs do not equal ReLU cost")
    return same_platform


def _directional_status(
    difference_pp: np.ndarray,
    *,
    delta_pp: float = RELEVANT_DIFFERENCE_PP,
    required_seed_agreement: int = REQUIRED_SEED_AGREEMENT,
) -> tuple[float, int, int, str]:
    mean = float(difference_pp.mean())
    positive = int((difference_pp > 0.0).sum())
    negative = int((difference_pp < 0.0).sum())
    if mean >= delta_pp and positive >= required_seed_agreement:
        status = "sustained"
    elif mean <= -delta_pp and negative >= required_seed_agreement:
        status = "refuted"
    else:
        status = "inconclusive"
    return mean, positive, negative, status


def evaluate_h3(
    final_validation_pct: Mapping[str, Sequence[float] | np.ndarray],
    relu_validation_pct: Sequence[float] | np.ndarray,
    parameters: Mapping[str, int],
    gflops: Mapping[str, float],
    returns: Mapping[str, float],
    *,
    relu_bridge_same_platform: bool | None = None,
) -> H3Evaluation:
    """Apply H3a–H3d exactly as frozen before the scientific runs."""
    values: dict[str, np.ndarray] = {}
    for config_id in CONFIG_ORDER:
        array = np.asarray(final_validation_pct[config_id], dtype=float)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError(f"{config_id} must contain three finite seed values")
        values[config_id] = array
    relu = np.asarray(relu_validation_pct, dtype=float)
    if relu.shape != (3,) or not np.isfinite(relu).all():
        raise ValueError("F-RELU must contain three finite seed values")

    comparisons: list[H3aComparison] = []
    for config_id in ("L2-IDENTITY", "L3-IDENTITY"):
        gain = values[config_id] - values["L1-DIRECT"]
        mean = float(gain.mean())
        positive = int((gain > 0.0).sum())
        negative = int((gain < 0.0).sum())
        contradicted = (
            mean >= RELEVANT_DIFFERENCE_PP
            and positive >= REQUIRED_SEED_AGREEMENT
        )
        comparisons.append(
            H3aComparison(
                config_id=config_id,
                mean_gain_pp=mean,
                positive_seeds=positive,
                negative_seeds=negative,
                status="contradicted" if contradicted else "not_contradicted",
            )
        )
    h3a_status = (
        "contradicted"
        if any(item.status == "contradicted" for item in comparisons)
        else "not_contradicted"
    )

    parameter_order = [int(parameters[item]) for item in CONFIG_ORDER]
    flop_order = [float(gflops[item]) for item in CONFIG_ORDER]
    h3b_status = (
        "sustained"
        if parameter_order[0] < parameter_order[1] < parameter_order[2]
        and flop_order[0] < flop_order[1] < flop_order[2]
        else "refuted"
    )

    return_order = [float(returns[item]) for item in CONFIG_ORDER]
    if return_order[0] > return_order[1] > return_order[2]:
        h3c_status = "sustained"
    elif return_order[0] < return_order[1] < return_order[2]:
        h3c_status = "refuted"
    else:
        h3c_status = "inconclusive"

    relu_difference = relu - values["L2-IDENTITY"]
    relu_mean, positive, negative, h3d_status = _directional_status(
        relu_difference
    )
    return H3Evaluation(
        h3a_comparisons=tuple(comparisons),
        h3a_status=h3a_status,
        h3b_status=h3b_status,
        h3c_status=h3c_status,
        h3d_status=h3d_status,
        relu_minus_l2_mean_pp=relu_mean,
        relu_minus_l2_positive_seeds=positive,
        relu_minus_l2_negative_seeds=negative,
        relu_bridge_same_platform=relu_bridge_same_platform,
    )


def aggregate_runs(
    grouped: Mapping[str, Sequence[RecordedRun]],
    relu_runs: Sequence[plot_v1.RecordedRun],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], H3Evaluation]:
    """Build curves, summary rows and the complete H3 decision."""
    same_platform = validate_relu_bridge(grouped, relu_runs)
    curves: dict[str, dict[str, Any]] = {}
    summary: list[dict[str, Any]] = []
    majority = next(iter(grouped.values()))[0].majority_accuracy
    metric_names = ("train_loss", "val_loss", "train_accuracy", "val_accuracy")

    for config_id in CONFIG_ORDER:
        runs = grouped[config_id]
        curve: dict[str, Any] = {"epochs": np.arange(1, 101)}
        for metric in metric_names:
            values = np.asarray(
                [[float(event[metric]) for event in run.epochs] for run in runs]
            )
            curve[f"{metric}_values"] = values
            curve[f"{metric}_mean"] = values.mean(axis=0)
            curve[f"{metric}_std"] = values.std(axis=0, ddof=1)
        val_values = curve["val_accuracy_values"]
        best_indices = val_values.argmax(axis=1)
        curve["best_epochs"] = best_indices + 1
        curve["best_val_accuracy_values"] = val_values[
            np.arange(3), best_indices
        ]
        curves[config_id] = curve

        flops = {int(run.row["flops_total"]) for run in runs}
        parameters = {int(run.row["parameters"]) for run in runs}
        if len(flops) != 1 or len(parameters) != 1:
            raise ValueError(f"cost changed between seeds for {config_id}")
        gflops = flops.pop() / 1e9
        mean_val = float(curve["val_accuracy_mean"][-1])
        row: dict[str, Any] = {
            "config_id": config_id,
            "depth": int(runs[0].row["depth"]),
            "parameters": parameters.pop(),
            "mean_train_loss": float(curve["train_loss_mean"][-1]),
            "mean_val_loss": float(curve["val_loss_mean"][-1]),
            "mean_train_accuracy_pct": float(
                curve["train_accuracy_mean"][-1] * 100
            ),
            "mean_val_accuracy_pct": mean_val * 100,
            "std_val_accuracy_pp": float(curve["val_accuracy_std"][-1] * 100),
            "mean_best_val_accuracy_pct": float(
                curve["best_val_accuracy_values"].mean() * 100
            ),
            "gflops_per_run": gflops,
            "return_pp_per_gflop": (mean_val - majority) * 100 / gflops,
        }
        for seed in range(3):
            row[f"best_val_accuracy_seed{seed}_pct"] = float(
                curve["best_val_accuracy_values"][seed] * 100
            )
            row[f"best_epoch_seed{seed}"] = int(curve["best_epochs"][seed])
        summary.append(row)

    for candidate in summary:
        dominated = any(
            other["mean_val_accuracy_pct"] >= candidate["mean_val_accuracy_pct"]
            and other["gflops_per_run"] <= candidate["gflops_per_run"]
            and (
                other["mean_val_accuracy_pct"] > candidate["mean_val_accuracy_pct"]
                or other["gflops_per_run"] < candidate["gflops_per_run"]
            )
            for other in summary
        )
        candidate["pareto_status"] = (
            "v3_dominated" if dominated else "v3_pareto"
        )

    ranked = sorted(
        summary,
        key=lambda item: item["mean_val_accuracy_pct"],
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["accuracy_rank"] = rank

    by_id = {row["config_id"]: row for row in summary}
    h3 = evaluate_h3(
        {
            config_id: curves[config_id]["val_accuracy_values"][:, -1] * 100
            for config_id in CONFIG_ORDER
        },
        [float(run.epochs[-1]["val_accuracy"]) * 100 for run in relu_runs],
        {config_id: by_id[config_id]["parameters"] for config_id in CONFIG_ORDER},
        {
            config_id: by_id[config_id]["gflops_per_run"]
            for config_id in CONFIG_ORDER
        },
        {
            config_id: by_id[config_id]["return_pp_per_gflop"]
            for config_id in CONFIG_ORDER
        },
        relu_bridge_same_platform=same_platform,
    )
    h3a_by_id = {
        comparison.config_id: comparison
        for comparison in h3.h3a_comparisons
    }
    for row in summary:
        row.update(
            {
                "h3a_status": h3.h3a_status,
                "h3a_l2_minus_l1_mean_pp": h3a_by_id[
                    "L2-IDENTITY"
                ].mean_gain_pp,
                "h3a_l2_positive_seeds": h3a_by_id[
                    "L2-IDENTITY"
                ].positive_seeds,
                "h3a_l2_negative_seeds": h3a_by_id[
                    "L2-IDENTITY"
                ].negative_seeds,
                "h3a_l3_minus_l1_mean_pp": h3a_by_id[
                    "L3-IDENTITY"
                ].mean_gain_pp,
                "h3a_l3_positive_seeds": h3a_by_id[
                    "L3-IDENTITY"
                ].positive_seeds,
                "h3a_l3_negative_seeds": h3a_by_id[
                    "L3-IDENTITY"
                ].negative_seeds,
                "h3b_status": h3.h3b_status,
                "h3c_status": h3.h3c_status,
                "h3d_status": h3.h3d_status,
                "h3d_relu_minus_l2_mean_pp": h3.relu_minus_l2_mean_pp,
                "h3d_positive_seeds": h3.relu_minus_l2_positive_seeds,
                "h3d_negative_seeds": h3.relu_minus_l2_negative_seeds,
                "relu_bridge_same_platform": h3.relu_bridge_same_platform,
            }
        )
    return curves, summary, h3


def write_summary(path: Path, summary: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=SUMMARY_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(
                {field: row[field] for field in SUMMARY_FIELDS}
                for row in summary
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _prepare_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_learning_curves(
    path: Path,
    curves: Mapping[str, Mapping[str, Any]],
) -> None:
    plt = _prepare_matplotlib()
    panels = (
        ("train_loss", "Loss de treino", "Loss", 1.0),
        ("val_loss", "Loss de validação", "Loss", 1.0),
        ("train_accuracy", "Acurácia de treino", "Acurácia (%)", 100.0),
        ("val_accuracy", "Acurácia de validação", "Acurácia (%)", 100.0),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    handles = []
    for axis, (metric, title, ylabel, scale) in zip(axes.flat, panels):
        for config_id in CONFIG_ORDER:
            curve = curves[config_id]
            epochs = curve["epochs"]
            values = curve[f"{metric}_values"] * scale
            mean = curve[f"{metric}_mean"] * scale
            std = curve[f"{metric}_std"] * scale
            for seed_values in values:
                axis.plot(
                    epochs,
                    seed_values,
                    color=COLORS[config_id],
                    linewidth=0.7,
                    alpha=0.15,
                )
            (line,) = axis.plot(
                epochs,
                mean,
                color=COLORS[config_id],
                linewidth=1.8,
                label=LABELS[config_id],
            )
            axis.fill_between(
                epochs,
                mean - std,
                mean + std,
                color=COLORS[config_id],
                alpha=0.12,
                linewidth=0,
            )
            if metric == "train_loss":
                handles.append(line)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks((1, 25, 50, 75, 100))
        axis.set_xlim(1, 100)
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Época")
    figure.suptitle(
        "V3 — média das seeds 0, 1 e 2 (faixa = 1 desvio-padrão)",
        y=0.995,
    )
    figure.legend(
        handles,
        [LABELS[item] for item in CONFIG_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_final_metrics(
    path: Path,
    curves: Mapping[str, Mapping[str, Any]],
) -> None:
    plt = _prepare_matplotlib()
    from matplotlib.lines import Line2D

    panels = (
        ("train_loss", "Loss de treino", "Loss", 1.0),
        ("val_loss", "Loss de validação", "Loss", 1.0),
        ("train_accuracy", "Acurácia de treino", "Acurácia (%)", 100.0),
        ("val_accuracy", "Acurácia de validação", "Acurácia (%)", 100.0),
    )
    positions = np.arange(len(CONFIG_ORDER))
    markers = ("o", "s", "^")
    offsets = (-0.12, 0.0, 0.12)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (metric, title, ylabel, scale) in zip(axes.flat, panels):
        for index, config_id in enumerate(CONFIG_ORDER):
            final_values = curves[config_id][f"{metric}_values"][:, -1] * scale
            for value, marker, offset in zip(final_values, markers, offsets):
                axis.scatter(
                    positions[index] + offset,
                    value,
                    marker=marker,
                    s=42,
                    color=COLORS[config_id],
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
            axis.errorbar(
                positions[index],
                final_values.mean(),
                yerr=final_values.std(ddof=1),
                fmt="D",
                markersize=6,
                capsize=5,
                color=COLORS[config_id],
                markeredgecolor="black",
                markeredgewidth=0.6,
                zorder=4,
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, [LABELS[item] for item in CONFIG_ORDER])
        axis.grid(axis="y", alpha=0.2)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            color="#666666",
            markerfacecolor="#666666",
            label=f"Seed {seed}",
        )
        for seed, marker in enumerate(markers)
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            color="#666666",
            markerfacecolor="#999999",
            markeredgecolor="black",
            label="Média ± desvio-padrão",
        )
    )
    figure.suptitle("V3 — métricas finais por seed na época 100", y=0.995)
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_accuracy_vs_flops(
    path: Path,
    summary: Sequence[dict[str, Any]],
    curves: Mapping[str, Mapping[str, Any]],
    relu_runs: Sequence[plot_v1.RecordedRun],
) -> None:
    plt = _prepare_matplotlib()
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for row in summary:
        config_id = row["config_id"]
        seed_values = curves[config_id]["val_accuracy_values"][:, -1] * 100
        for seed, value in enumerate(seed_values):
            axis.scatter(
                row["gflops_per_run"],
                value,
                s=34,
                marker=("o", "s", "^")[seed],
                facecolor="none",
                edgecolor=COLORS[config_id],
                linewidth=0.9,
                alpha=0.75,
            )
        axis.errorbar(
            row["gflops_per_run"],
            row["mean_val_accuracy_pct"],
            yerr=row["std_val_accuracy_pp"],
            fmt="D",
            markersize=8,
            capsize=5,
            color=COLORS[config_id],
            label=LABELS[config_id],
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        axis.annotate(
            f"{int(row['parameters']):,} params".replace(",", "."),
            (row["gflops_per_run"], row["mean_val_accuracy_pct"]),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=8,
        )
    pareto_rows = sorted(
        (
            row for row in summary
            if row["pareto_status"] == "v3_pareto"
        ),
        key=lambda row: row["gflops_per_run"],
    )
    if len(pareto_rows) > 1:
        axis.plot(
            [row["gflops_per_run"] for row in pareto_rows],
            [row["mean_val_accuracy_pct"] for row in pareto_rows],
            color="#666666",
            linestyle="--",
            linewidth=1,
            label="Pareto somente V3",
        )
    relu_gflops = int(relu_runs[0].row["flops_total"]) / 1e9
    relu_values = np.asarray(
        [float(run.epochs[-1]["val_accuracy"]) * 100 for run in relu_runs]
    )
    axis.errorbar(
        relu_gflops,
        float(relu_values.mean()),
        yerr=float(relu_values.std(ddof=1)),
        marker="*",
        markersize=14,
        capsize=5,
        color="#2ca02c",
        label="F-RELU (V1)",
    )
    axis.set(
        xlabel="GFLOPs instrumentados por run",
        ylabel="Acurácia média de validação (%)",
        title="V3 — desempenho versus custo (Pareto calculada só na V3)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def generate(
    *,
    results_path: Path = DEFAULT_RESULTS,
    v1_results_path: Path = DEFAULT_V1_RESULTS,
    summary_path: Path = DEFAULT_SUMMARY,
    plots_dir: Path = DEFAULT_PLOTS,
) -> tuple[list[dict[str, Any]], H3Evaluation]:
    grouped = load_v3_runs(results_path)
    relu_runs = load_relu_reference(v1_results_path)
    curves, summary, h3 = aggregate_runs(grouped, relu_runs)
    write_summary(summary_path, summary)
    plot_learning_curves(plots_dir / "learning_curves.png", curves)
    plot_final_metrics(plots_dir / "final_metrics_by_seed.png", curves)
    plot_accuracy_vs_flops(
        plots_dir / "accuracy_vs_flops.png",
        summary,
        curves,
        relu_runs,
    )
    return summary, h3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate Variable 3 summary and plots.",
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--v1-results", type=Path, default=DEFAULT_V1_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS)
    return parser


def main(argv: Sequence[str] | None = None) -> tuple[list[dict[str, Any]], H3Evaluation]:
    args = _build_parser().parse_args(argv)
    summary, h3 = generate(
        results_path=args.results,
        v1_results_path=args.v1_results,
        summary_path=args.summary,
        plots_dir=args.plots_dir,
    )
    print(
        "V3 analysis: "
        f"H3a={h3.h3a_status}, H3b={h3.h3b_status}, "
        f"H3c={h3.h3c_status}, H3d={h3.h3d_status}"
    )
    return summary, h3


if __name__ == "__main__":
    main()
