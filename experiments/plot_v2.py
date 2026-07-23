"""Regenerate the Variable 2 summary and plots from recorded V2 artifacts.

The primary analysis always uses validation accuracy at epoch 100. Per-seed
best epochs are recorded only as a secondary description of convergence.
"""

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

from experiments import run_v2


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = Path(run_v2.DEFAULT_ARTIFACTS)
DEFAULT_RESULTS = DEFAULT_ARTIFACTS / "results.csv"
DEFAULT_SUMMARY = DEFAULT_ARTIFACTS / "summary.csv"
DEFAULT_PLOTS = DEFAULT_ARTIFACTS / "plots"

CONFIG_ORDER = ("S-BETA-0.5", "S-BETA-1", "S-BETA-2", "S-BETA-5")
BETAS = {
    "S-BETA-0.5": 0.5,
    "S-BETA-1": 1.0,
    "S-BETA-2": 2.0,
    "S-BETA-5": 5.0,
}
LABELS = {
    config_id: f"β={beta:g}".replace(".", ",")
    for config_id, beta in BETAS.items()
}
COLORS = {
    "S-BETA-0.5": "#9467bd",
    "S-BETA-1": "#1f77b4",
    "S-BETA-2": "#2ca02c",
    "S-BETA-5": "#d62728",
}
CENTRAL_CONFIGS = ("S-BETA-1", "S-BETA-2")
EXTREME_CONFIGS = ("S-BETA-0.5", "S-BETA-5")
DIAGNOSTIC_EPOCHS = (0, 1, 25, 50, 75, 100)
RELEVANT_DIFFERENCE_PP = 0.5
REQUIRED_SEED_AGREEMENT = 2
H2_LABELS = {
    "sustained": "sustentada",
    "refuted": "refutada",
    "inconclusive": "inconclusiva",
}

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
    "parameters",
)

SUMMARY_FIELDS = (
    "accuracy_rank",
    "config_id",
    "beta",
    "h2_group",
    "group_winner",
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
    "std_best_val_accuracy_pp",
    "gflops_per_run",
    "return_pp_per_gflop",
    "h2_status",
    "h2_central_winners",
    "h2_extreme_winners",
)


@dataclass(frozen=True)
class RecordedRun:
    """One validated V2 scientific run."""

    row: dict[str, str]
    epochs: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]
    majority_accuracy: float


@dataclass(frozen=True)
class H2Comparison:
    """One paired comparison between a central and an extreme co-winner."""

    central_config: str
    extreme_config: str
    mean_difference_pp: float
    positive_seeds: int
    negative_seeds: int
    status: str


@dataclass(frozen=True)
class H2Evaluation:
    """Complete H2 decision, including the frozen tie policy."""

    central_winners: tuple[str, ...]
    extreme_winners: tuple[str, ...]
    comparisons: tuple[H2Comparison, ...]
    status: str


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
    if not run_v2.result_artifacts_valid(row):
        raise ValueError(f"invalid artifacts for {row['run_id']}")
    if row["test_accuracy"].strip():
        raise ValueError("V2 analysis must not consume official-test accuracy")

    events = _read_events(_path_from_row(row["log_path"]))
    started = [event for event in events if event.get("event") == "run_started"]
    epochs = tuple(event for event in events if event.get("event") == "epoch")
    diagnostics = tuple(
        event for event in events if event.get("event") == "diagnostic"
    )
    if len(started) != 1:
        raise ValueError(f"run_started is missing or duplicated in {row['run_id']}")
    start_data = started[0].get("data", {})
    if start_data.get("official_test_loaded") is not False:
        raise ValueError(f"official-test policy violated in {row['run_id']}")
    if [event.get("epoch") for event in epochs] != list(range(1, 101)):
        raise ValueError(f"incomplete epoch history in {row['run_id']}")
    if [event.get("epoch") for event in diagnostics] != list(DIAGNOSTIC_EPOCHS):
        raise ValueError(f"incomplete diagnostics in {row['run_id']}")

    flops_per_epoch = int(row["flops_per_epoch"])
    if any(int(event.get("flops", -1)) != flops_per_epoch for event in epochs):
        raise ValueError(f"epoch FLOPs disagree with results.csv in {row['run_id']}")
    if int(row["flops_total"]) != 100 * flops_per_epoch:
        raise ValueError(f"total FLOPs disagree with epochs in {row['run_id']}")

    final = epochs[-1]
    final_pairs = {
        "train_loss_final": "train_loss",
        "val_loss_final": "val_loss",
        "train_accuracy": "train_accuracy",
        "val_accuracy": "val_accuracy",
    }
    for field, event_field in final_pairs.items():
        if float(row[field]) != float(final[event_field]):
            raise ValueError(f"final metric mismatch for {field} in {row['run_id']}")
    metric_values = np.array(
        [
            float(event[field])
            for event in epochs
            for field in (
                "train_loss",
                "val_loss",
                "train_accuracy",
                "val_accuracy",
            )
        ]
    )
    if not np.isfinite(metric_values).all():
        raise ValueError(f"non-finite metric in {row['run_id']}")

    try:
        majority_accuracy = float(start_data["validation_majority_accuracy"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"validation majority accuracy missing in {row['run_id']}"
        ) from error
    if not np.isfinite(majority_accuracy):
        raise ValueError(f"invalid validation majority accuracy in {row['run_id']}")

    return RecordedRun(
        row=row,
        epochs=epochs,
        diagnostics=diagnostics,
        majority_accuracy=majority_accuracy,
    )


def load_v2_runs(
    results_path: Path = DEFAULT_RESULTS,
) -> dict[str, tuple[RecordedRun, ...]]:
    """Load and strictly validate the complete 4-beta by 3-seed V2 grid."""

    rows = run_v2.load_results(results_path)
    if len(rows) != 12:
        raise ValueError("V2 must contain exactly 12 scientific rows")
    if any(row["variable"] != "V2_softplus_curvature" for row in rows):
        raise ValueError("V2 results contain a row from another variable")
    if any(
        row["status"] != "completed_valid"
        or row["run_kind"] != "scientific"
        or row["purpose"] != "primary"
        or int(row["epochs"]) != 100
        for row in rows
    ):
        raise ValueError(
            "all V2 rows must be valid 100-epoch primary scientific runs"
        )
    if any(row["config_id"] not in CONFIG_ORDER for row in rows):
        raise ValueError("V2 contains an unknown configuration")

    for field in CONTEXT_FIELDS:
        if len({row[field] for row in rows}) != 1:
            raise ValueError(f"V2 runs mix experimental contexts: {field}")

    grouped: dict[str, tuple[RecordedRun, ...]] = {}
    for config_id in CONFIG_ORDER:
        selected = sorted(
            (row for row in rows if row["config_id"] == config_id),
            key=lambda row: int(row["seed"]),
        )
        if [int(row["seed"]) for row in selected] != [0, 1, 2]:
            raise ValueError(f"{config_id} does not contain seeds 0, 1 and 2")
        for row in selected:
            seed = int(row["seed"])
            if row["activation"] != "softplus_beta":
                raise ValueError(f"unexpected activation in {row['run_id']}")
            if float(row["beta"]) != BETAS[config_id]:
                raise ValueError(f"recorded beta differs from {config_id}")
            expected_id = run_v2.expected_run_id(config_id, seed)
            if row["run_id"] != expected_id or int(row["repetition"]) != 1:
                raise ValueError(f"unexpected run identity: {row['run_id']}")
        grouped[config_id] = tuple(
            _load_recorded_run(row) for row in selected
        )

    for seed in (0, 1, 2):
        hashes = {
            run.row["initial_weights_hash"]
            for runs in grouped.values()
            for run in runs
            if int(run.row["seed"]) == seed
        }
        if len(hashes) != 1:
            raise ValueError(
                f"initial weights differ between beta levels for seed {seed}"
            )

    majority = {run.majority_accuracy for runs in grouped.values() for run in runs}
    if len(majority) != 1:
        raise ValueError("validation majority accuracy changed between V2 runs")
    costs = {
        (
            int(run.row["flops_per_epoch"]),
            int(run.row["flops_total"]),
            int(run.row["inference_flops_total"]),
        )
        for runs in grouped.values()
        for run in runs
    }
    if len(costs) != 1:
        raise ValueError("Softplus-beta levels do not have identical FLOPs")
    return grouped


def _group_winners(
    means: Mapping[str, float],
    candidates: Sequence[str],
) -> tuple[str, ...]:
    best = max(means[config_id] for config_id in candidates)
    return tuple(config_id for config_id in candidates if means[config_id] == best)


def _classify_h2_pair(
    central: np.ndarray,
    extreme: np.ndarray,
    *,
    delta_pp: float,
    required_seed_agreement: int,
) -> tuple[float, int, int, str]:
    difference = central - extreme
    mean_difference = float(difference.mean())
    positive_seeds = int((difference > 0.0).sum())
    negative_seeds = int((difference < 0.0).sum())
    if (
        mean_difference >= delta_pp
        and positive_seeds >= required_seed_agreement
    ):
        status = "sustained"
    elif (
        mean_difference <= -delta_pp
        and negative_seeds >= required_seed_agreement
    ):
        status = "refuted"
    else:
        status = "inconclusive"
    return mean_difference, positive_seeds, negative_seeds, status


def evaluate_h2(
    final_validation_pct: Mapping[str, Sequence[float] | np.ndarray],
    *,
    delta_pp: float = RELEVANT_DIFFERENCE_PP,
    required_seed_agreement: int = REQUIRED_SEED_AGREEMENT,
) -> H2Evaluation:
    """Apply H2 exactly, including paired seeds and exact co-winner ties."""

    if delta_pp <= 0.0:
        raise ValueError("delta_pp must be positive")
    if required_seed_agreement not in (1, 2, 3):
        raise ValueError("required_seed_agreement must be between 1 and 3")

    values: dict[str, np.ndarray] = {}
    for config_id in CONFIG_ORDER:
        try:
            array = np.asarray(final_validation_pct[config_id], dtype=float)
        except KeyError as error:
            raise ValueError(f"missing final accuracies for {config_id}") from error
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError(f"{config_id} must contain three finite seed values")
        values[config_id] = array

    means = {config_id: float(array.mean()) for config_id, array in values.items()}
    central_winners = _group_winners(means, CENTRAL_CONFIGS)
    extreme_winners = _group_winners(means, EXTREME_CONFIGS)
    comparisons: list[H2Comparison] = []
    for central_id in central_winners:
        for extreme_id in extreme_winners:
            mean_difference, positive, negative, status = _classify_h2_pair(
                values[central_id],
                values[extreme_id],
                delta_pp=delta_pp,
                required_seed_agreement=required_seed_agreement,
            )
            comparisons.append(
                H2Comparison(
                    central_config=central_id,
                    extreme_config=extreme_id,
                    mean_difference_pp=mean_difference,
                    positive_seeds=positive,
                    negative_seeds=negative,
                    status=status,
                )
            )

    pair_statuses = {comparison.status for comparison in comparisons}
    status = pair_statuses.pop() if len(pair_statuses) == 1 else "inconclusive"
    return H2Evaluation(
        central_winners=central_winners,
        extreme_winners=extreme_winners,
        comparisons=tuple(comparisons),
        status=status,
    )


def aggregate_runs(
    grouped: Mapping[str, Sequence[RecordedRun]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], H2Evaluation]:
    """Aggregate final metrics, raw curves, secondary best epochs and H2."""

    if set(grouped) != set(CONFIG_ORDER) or len(grouped) != len(CONFIG_ORDER):
        raise ValueError("V2 groups differ from the frozen configuration grid")
    if any(len(grouped[config_id]) != 3 for config_id in CONFIG_ORDER):
        raise ValueError("each V2 configuration must contain exactly three runs")
    majority = grouped[CONFIG_ORDER[0]][0].majority_accuracy
    if any(
        run.majority_accuracy != majority
        for config_id in CONFIG_ORDER
        for run in grouped[config_id]
    ):
        raise ValueError("validation majority accuracy changed between V2 runs")
    metric_names = ("train_loss", "val_loss", "train_accuracy", "val_accuracy")
    curves: dict[str, dict[str, Any]] = {}

    for config_id in CONFIG_ORDER:
        runs = tuple(grouped[config_id])
        curve: dict[str, Any] = {"epochs": np.arange(1, 101)}
        for metric in metric_names:
            values = np.array(
                [[float(event[metric]) for event in run.epochs] for run in runs],
                dtype=float,
            )
            if values.shape != (3, 100) or not np.isfinite(values).all():
                raise ValueError(f"invalid {metric} history for {config_id}")
            curve[f"{metric}_values"] = values
            curve[f"{metric}_mean"] = values.mean(axis=0)
            curve[f"{metric}_std"] = values.std(axis=0, ddof=1)
        best_indices = np.argmax(curve["val_accuracy_values"], axis=1)
        curve["best_epochs"] = best_indices + 1
        curve["best_val_accuracy_values"] = curve["val_accuracy_values"][
            np.arange(3), best_indices
        ]
        curves[config_id] = curve

    final_validation_pct = {
        config_id: curves[config_id]["val_accuracy_values"][:, -1] * 100.0
        for config_id in CONFIG_ORDER
    }
    h2 = evaluate_h2(final_validation_pct)
    winners = set(h2.central_winners) | set(h2.extreme_winners)
    central_winners = "|".join(h2.central_winners)
    extreme_winners = "|".join(h2.extreme_winners)

    summary: list[dict[str, Any]] = []
    for config_id in CONFIG_ORDER:
        runs = tuple(grouped[config_id])
        curve = curves[config_id]
        costs = {int(run.row["flops_total"]) for run in runs}
        if len(costs) != 1:
            raise ValueError(f"FLOPs differ between seeds for {config_id}")
        gflops = costs.pop() / 1e9
        final_validation = curve["val_accuracy_values"][:, -1]
        best_validation = curve["best_val_accuracy_values"]
        item: dict[str, Any] = {
            "config_id": config_id,
            "beta": BETAS[config_id],
            "h2_group": "central" if config_id in CENTRAL_CONFIGS else "extreme",
            "group_winner": "yes" if config_id in winners else "no",
            "mean_train_loss": float(curve["train_loss_mean"][-1]),
            "mean_val_loss": float(curve["val_loss_mean"][-1]),
            "mean_train_accuracy_pct": float(
                curve["train_accuracy_mean"][-1] * 100.0
            ),
            "mean_val_accuracy_pct": float(final_validation.mean() * 100.0),
            "std_val_accuracy_pp": float(final_validation.std(ddof=1) * 100.0),
            "mean_best_val_accuracy_pct": float(best_validation.mean() * 100.0),
            "std_best_val_accuracy_pp": float(
                best_validation.std(ddof=1) * 100.0
            ),
            "gflops_per_run": gflops,
            "return_pp_per_gflop": (
                float(final_validation.mean() - majority) * 100.0 / gflops
            ),
            "h2_status": h2.status,
            "h2_central_winners": central_winners,
            "h2_extreme_winners": extreme_winners,
        }
        for seed in range(3):
            item[f"best_val_accuracy_seed{seed}_pct"] = float(
                best_validation[seed] * 100.0
            )
            item[f"best_epoch_seed{seed}"] = int(curve["best_epochs"][seed])
        summary.append(item)

    ranked = sorted(
        summary,
        key=lambda item: (-item["mean_val_accuracy_pct"], item["beta"]),
    )
    for rank, item in enumerate(ranked, start=1):
        item["accuracy_rank"] = rank
    return curves, summary, h2


def write_summary(path: Path, summary: Sequence[Mapping[str, Any]]) -> None:
    """Write an atomic, stable and LF-terminated aggregate table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=SUMMARY_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            for item in sorted(summary, key=lambda value: value["accuracy_rank"]):
                writer.writerow(
                    {
                        field: (
                            f"{item[field]:.10f}"
                            if isinstance(item[field], (float, np.floating))
                            else item[field]
                        )
                        for field in SUMMARY_FIELDS
                    }
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

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def plot_learning_curves(path: Path, curves: Mapping[str, Mapping[str, Any]]) -> None:
    """Plot raw seeds, mean and one sample SD for the four training metrics."""

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
    for axis in axes[-1]:
        axis.set_xlabel("Época")
    figure.suptitle(
        "Variável 2 — média das seeds 0, 1 e 2 (faixa = 1 desvio-padrão)",
        y=0.995,
    )
    figure.legend(
        handles,
        [LABELS[item] for item in CONFIG_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
        metadata={"Software": "BERT-cpu experiments.plot_v2"},
    )
    plt.close(figure)


def plot_final_metrics(
    path: Path,
    curves: Mapping[str, Mapping[str, Any]],
) -> None:
    """Show each seed and mean ± sample SD for metrics at epoch 100."""

    plt = _prepare_matplotlib()
    from matplotlib.lines import Line2D

    panels = (
        ("train_loss", "Loss de treino", "Loss", 1.0),
        ("val_loss", "Loss de validação", "Loss", 1.0),
        ("train_accuracy", "Acurácia de treino", "Acurácia (%)", 100.0),
        ("val_accuracy", "Acurácia de validação", "Acurácia (%)", 100.0),
    )
    markers = ("o", "s", "^")
    offsets = (-0.12, 0.0, 0.12)
    positions = np.arange(len(CONFIG_ORDER), dtype=float)
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
    figure.suptitle("Variável 2 — métricas finais por seed na época 100", y=0.995)
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
        metadata={"Software": "BERT-cpu experiments.plot_v2"},
    )
    plt.close(figure)


def plot_validation_vs_beta(
    path: Path,
    curves: Mapping[str, Mapping[str, Any]],
    h2: H2Evaluation,
) -> None:
    """Plot final validation accuracy as a direct function of fixed beta."""

    plt = _prepare_matplotlib()
    figure, axis = plt.subplots(figsize=(8, 5.5))
    beta_values = np.array([BETAS[config_id] for config_id in CONFIG_ORDER])
    means = np.array(
        [
            curves[config_id]["val_accuracy_values"][:, -1].mean() * 100.0
            for config_id in CONFIG_ORDER
        ]
    )
    standard_deviations = np.array(
        [
            curves[config_id]["val_accuracy_values"][:, -1].std(ddof=1) * 100.0
            for config_id in CONFIG_ORDER
        ]
    )
    axis.plot(beta_values, means, color="#555555", linewidth=1.2, zorder=1)
    seed_markers = ("o", "s", "^")
    offsets = (-0.035, 0.0, 0.035)
    for index, config_id in enumerate(CONFIG_ORDER):
        seed_values = curves[config_id]["val_accuracy_values"][:, -1] * 100.0
        for value, marker, offset in zip(seed_values, seed_markers, offsets):
            axis.scatter(
                beta_values[index] + offset,
                value,
                marker=marker,
                s=34,
                facecolor="none",
                edgecolor=COLORS[config_id],
                linewidth=0.9,
                zorder=2,
            )
        axis.errorbar(
            beta_values[index],
            means[index],
            yerr=standard_deviations[index],
            fmt="D",
            markersize=7,
            capsize=5,
            color=COLORS[config_id],
            markeredgecolor="black",
            markeredgewidth=0.6,
            zorder=3,
        )
    axis.set_xticks(beta_values, [f"{value:g}" for value in beta_values])
    axis.set_xlabel("Curvatura fixa β")
    axis.set_ylabel("Acurácia de validação na época 100 (%)")
    axis.set_title(
        "Variável 2 — validação versus β "
        f"(média ± desvio-padrão; H2: {H2_LABELS[h2.status]})"
    )
    axis.text(
        0.01,
        0.01,
        "Marcadores vazios: seeds individuais",
        transform=axis.transAxes,
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
        metadata={"Software": "BERT-cpu experiments.plot_v2"},
    )
    plt.close(figure)


def generate(
    *,
    results_path: Path = DEFAULT_RESULTS,
    summary_path: Path = DEFAULT_SUMMARY,
    plots_dir: Path = DEFAULT_PLOTS,
) -> tuple[list[dict[str, Any]], H2Evaluation]:
    """Validate all evidence and regenerate the complete V2 visual summary."""

    grouped = load_v2_runs(results_path)
    curves, summary, h2 = aggregate_runs(grouped)
    write_summary(summary_path, summary)
    plot_learning_curves(plots_dir / "learning_curves.png", curves)
    plot_final_metrics(plots_dir / "final_metrics_by_seed.png", curves)
    plot_validation_vs_beta(plots_dir / "validation_vs_beta.png", curves, h2)
    return (
        sorted(summary, key=lambda item: item["accuracy_rank"]),
        h2,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate final Variable 2 plots.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS)
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], H2Evaluation]:
    args = _build_parser().parse_args(argv)
    summary, h2 = generate(
        results_path=args.results.resolve(),
        summary_path=args.summary.resolve(),
        plots_dir=args.plots_dir.resolve(),
    )
    for item in summary:
        print(
            f"{item['accuracy_rank']}. beta={item['beta']:g}: "
            f"val={item['mean_val_accuracy_pct']:.4f}% | "
            f"FLOPs={item['gflops_per_run']:.7f} GFLOPs"
        )
    for comparison in h2.comparisons:
        print(
            f"H2 {comparison.central_config} - {comparison.extreme_config}: "
            f"{comparison.mean_difference_pp:+.4f} p.p. | "
            f"seeds +/−={comparison.positive_seeds}/{comparison.negative_seeds} | "
            f"{comparison.status}"
        )
    print(f"H2 final: {h2.status}")
    return summary, h2


if __name__ == "__main__":
    main()
