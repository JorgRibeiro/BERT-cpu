"""Regenerate the final Variable 1 tables and plots from recorded artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from experiments import run_v1


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "experiments/results.csv"
DEFAULT_SUMMARY = REPO_ROOT / "experiments/v1_summary.csv"
DEFAULT_PLOTS = REPO_ROOT / "experiments/plots"

CONFIG_ORDER = ("F-RELU", "F-SIGMOID", "F-SWISH", "F-SOFTPLUS")
LABELS = {
    "F-RELU": "ReLU",
    "F-SIGMOID": "Sigmoid",
    "F-SWISH": "Swish",
    "F-SOFTPLUS": "Softplus",
}
COLORS = {
    "F-RELU": "#1f77b4",
    "F-SIGMOID": "#d62728",
    "F-SWISH": "#2ca02c",
    "F-SOFTPLUS": "#ff7f0e",
}
DIAGNOSTIC_EPOCHS = (0, 1, 25, 50, 75, 100)
CONTEXT_FIELDS = (
    "task",
    "variable",
    "run_kind",
    "phase",
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
    "activation",
    "mean_train_loss",
    "mean_val_loss",
    "mean_train_accuracy_pct",
    "mean_val_accuracy_pct",
    "std_val_accuracy_pp",
    "gflops_per_run",
    "return_pp_per_gflop",
    "pareto_status",
)


@dataclass(frozen=True)
class RecordedRun:
    row: dict[str, str]
    epochs: tuple[dict[str, Any], ...]
    diagnostics: tuple[dict[str, Any], ...]
    majority_accuracy: float


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
    if not run_v1._result_artifacts_valid(row):
        raise ValueError(f"invalid artifacts for {row['run_id']}")
    if row["test_accuracy"]:
        raise ValueError("V1 train/validation plots must not consume test accuracy")

    events = _read_events(_path_from_row(row["log_path"]))
    started = [event for event in events if event["event"] == "run_started"]
    epochs = tuple(event for event in events if event["event"] == "epoch")
    diagnostics = tuple(event for event in events if event["event"] == "diagnostic")
    if len(started) != 1 or started[0]["data"]["official_test_loaded"] is not False:
        raise ValueError(f"official-test policy violated in {row['run_id']}")
    if [event["epoch"] for event in epochs] != list(range(1, 101)):
        raise ValueError(f"incomplete epoch history in {row['run_id']}")
    if [event["epoch"] for event in diagnostics] != list(DIAGNOSTIC_EPOCHS):
        raise ValueError(f"incomplete diagnostics in {row['run_id']}")
    if any(event["flops"] != int(row["flops_per_epoch"]) for event in epochs):
        raise ValueError(f"epoch FLOPs disagree with results.csv in {row['run_id']}")

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

    return RecordedRun(
        row=row,
        epochs=epochs,
        diagnostics=diagnostics,
        majority_accuracy=float(started[0]["data"]["validation_majority_accuracy"]),
    )


def load_v1_runs(results_path: Path = DEFAULT_RESULTS) -> dict[str, tuple[RecordedRun, ...]]:
    """Load and strictly validate the 12 primary V1 runs."""
    with results_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != run_v1.RESULT_FIELDS:
            raise ValueError("results.csv header differs from the runner schema")
        all_rows = list(reader)

    rows = [row for row in all_rows if row["variable"] == "V1_activation_family"]
    if len(rows) != 13 or any(row["status"] != "completed_valid" for row in rows):
        raise ValueError("V1 must contain exactly 13 valid rows")
    if any(row["config_id"] not in CONFIG_ORDER for row in rows):
        raise ValueError("V1 contains an unknown configuration")

    repetitions = [row for row in rows if row["purpose"] == "determinism_repeat"]
    if len(repetitions) != 1 or (
        repetitions[0]["config_id"],
        repetitions[0]["seed"],
        repetitions[0]["repetition"],
    ) != ("F-RELU", "0", "2"):
        raise ValueError("the single ReLU determinism repeat is missing or ambiguous")
    _load_recorded_run(repetitions[0])

    primary_rows = [row for row in rows if row["purpose"] == "primary"]
    if len(primary_rows) != 12:
        raise ValueError("V1 must contain exactly 12 primary rows")
    for field in CONTEXT_FIELDS:
        if len({row[field] for row in primary_rows}) != 1:
            raise ValueError(f"primary runs mix experimental contexts: {field}")
    for seed in (0, 1, 2):
        hashes = {
            row["initial_weights_hash"]
            for row in primary_rows
            if int(row["seed"]) == seed
        }
        if len(hashes) != 1:
            raise ValueError(f"initial weights differ across activations for seed {seed}")

    grouped: dict[str, tuple[RecordedRun, ...]] = {}
    for config_id in CONFIG_ORDER:
        selected = sorted(
            (row for row in primary_rows if row["config_id"] == config_id),
            key=lambda row: int(row["seed"]),
        )
        if [int(row["seed"]) for row in selected] != [0, 1, 2]:
            raise ValueError(f"{config_id} does not contain seeds 0, 1 and 2")
        grouped[config_id] = tuple(_load_recorded_run(row) for row in selected)

    majority = {run.majority_accuracy for runs in grouped.values() for run in runs}
    if len(majority) != 1:
        raise ValueError("validation majority accuracy changed between runs")
    return grouped


def aggregate_runs(
    grouped: dict[str, tuple[RecordedRun, ...]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Return epoch curves and the final summary for each activation."""
    curves: dict[str, dict[str, Any]] = {}
    summary: list[dict[str, Any]] = []
    majority = next(iter(grouped.values()))[0].majority_accuracy
    metric_names = ("train_loss", "val_loss", "train_accuracy", "val_accuracy")

    for config_id in CONFIG_ORDER:
        runs = grouped[config_id]
        curve: dict[str, Any] = {"epochs": np.arange(1, 101)}
        for metric in metric_names:
            values = np.array(
                [[float(event[metric]) for event in run.epochs] for run in runs]
            )
            curve[f"{metric}_values"] = values
            curve[f"{metric}_mean"] = values.mean(axis=0)
            curve[f"{metric}_std"] = values.std(axis=0, ddof=1)
        curves[config_id] = curve

        flops = {int(run.row["flops_total"]) for run in runs}
        if len(flops) != 1:
            raise ValueError(f"FLOPs differ between seeds for {config_id}")
        gflops = flops.pop() / 1e9
        mean_val = float(curve["val_accuracy_mean"][-1])
        summary.append(
            {
                "config_id": config_id,
                "activation": LABELS[config_id],
                "mean_train_loss": float(curve["train_loss_mean"][-1]),
                "mean_val_loss": float(curve["val_loss_mean"][-1]),
                "mean_train_accuracy_pct": float(curve["train_accuracy_mean"][-1] * 100),
                "mean_val_accuracy_pct": mean_val * 100,
                "std_val_accuracy_pp": float(curve["val_accuracy_std"][-1] * 100),
                "gflops_per_run": gflops,
                "return_pp_per_gflop": (mean_val - majority) * 100 / gflops,
            }
        )

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
        candidate["pareto_status"] = "dominated" if dominated else "pareto"

    ranked = sorted(summary, key=lambda item: item["mean_val_accuracy_pct"], reverse=True)
    for rank, item in enumerate(ranked, start=1):
        item["accuracy_rank"] = rank
    return curves, summary


def write_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    """Write a stable, LF-terminated aggregate table."""
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
                            if isinstance(item[field], float)
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


def plot_learning_curves(path: Path, curves: dict[str, dict[str, Any]]) -> None:
    """Plot mean and sample standard deviation across the three seeds."""
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
        "Variável 1 — média das seeds 0, 1 e 2 (faixa = 1 desvio-padrão)",
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
        metadata={"Software": "BERT-cpu experiments.plot_v1"},
    )
    plt.close(figure)


def plot_final_metrics(path: Path, curves: dict[str, dict[str, Any]]) -> None:
    """Show every seed plus mean and sample standard deviation at epoch 100."""
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
            for seed, (value, marker, offset) in enumerate(
                zip(final_values, markers, offsets)
            ):
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
        axis.tick_params(axis="x", rotation=12)
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
    figure.suptitle("Variável 1 — métricas finais por seed na época 100", y=0.995)
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
        metadata={"Software": "BERT-cpu experiments.plot_v1"},
    )
    plt.close(figure)


def plot_accuracy_vs_flops(
    path: Path,
    summary: list[dict[str, Any]],
    curves: dict[str, dict[str, Any]],
) -> None:
    """Plot the final validation accuracy/cost pairs and mean-level Pareto front."""
    plt = _prepare_matplotlib()
    figure, axis = plt.subplots(figsize=(8, 5.5))
    by_id = {item["config_id"]: item for item in summary}
    pareto = sorted(
        (item for item in summary if item["pareto_status"] == "pareto"),
        key=lambda item: item["gflops_per_run"],
    )
    relu_budget = by_id["F-RELU"]["gflops_per_run"]
    axis.axvline(
        relu_budget,
        color=COLORS["F-RELU"],
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
        label="Orçamento fixo da ReLU",
        zorder=0,
    )
    axis.plot(
        [item["gflops_per_run"] for item in pareto],
        [item["mean_val_accuracy_pct"] for item in pareto],
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="Fronteira de Pareto",
        zorder=1,
    )
    for config_id in CONFIG_ORDER:
        item = by_id[config_id]
        pareto_point = item["pareto_status"] == "pareto"
        seed_values = curves[config_id]["val_accuracy_values"][:, -1] * 100
        for seed, (value, marker) in enumerate(zip(seed_values, ("o", "s", "^"))):
            axis.scatter(
                item["gflops_per_run"],
                value,
                marker=marker,
                s=25,
                facecolor="none",
                edgecolor=COLORS[config_id],
                linewidth=0.8,
                alpha=0.75,
                zorder=2,
            )
        axis.errorbar(
            item["gflops_per_run"],
            item["mean_val_accuracy_pct"],
            yerr=item["std_val_accuracy_pp"],
            fmt="o" if pareto_point else "X",
            markersize=8,
            capsize=4,
            color=COLORS[config_id],
            label=f"{LABELS[config_id]} ({'Pareto' if pareto_point else 'dominada'})",
            zorder=3,
        )
        axis.annotate(
            LABELS[config_id],
            (item["gflops_per_run"], item["mean_val_accuracy_pct"]),
            xytext=(6, 7),
            textcoords="offset points",
            color=COLORS[config_id],
            fontweight="bold",
        )
    axis.set_xlabel("FLOPs instrumentados por run (GFLOPs)")
    axis.set_ylabel("Acurácia de validação na época 100 (%)")
    axis.set_title("Variável 1 — desempenho versus custo (média ± desvio-padrão)")
    axis.text(
        0.01,
        0.01,
        "Marcadores vazios: seeds individuais",
        transform=axis.transAxes,
        fontsize=8,
        color="#555555",
    )
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
        metadata={"Software": "BERT-cpu experiments.plot_v1"},
    )
    plt.close(figure)


def generate(
    *,
    results_path: Path = DEFAULT_RESULTS,
    summary_path: Path = DEFAULT_SUMMARY,
    plots_dir: Path = DEFAULT_PLOTS,
) -> list[dict[str, Any]]:
    grouped = load_v1_runs(results_path)
    curves, summary = aggregate_runs(grouped)
    write_summary(summary_path, summary)
    plot_learning_curves(plots_dir / "v1_learning_curves.png", curves)
    plot_final_metrics(plots_dir / "v1_final_metrics_by_seed.png", curves)
    plot_accuracy_vs_flops(
        plots_dir / "v1_accuracy_vs_flops.png",
        summary,
        curves,
    )
    return sorted(summary, key=lambda item: item["accuracy_rank"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate final Variable 1 plots.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS)
    return parser


def main(argv: Sequence[str] | None = None) -> list[dict[str, Any]]:
    args = _build_parser().parse_args(argv)
    summary = generate(
        results_path=args.results.resolve(),
        summary_path=args.summary.resolve(),
        plots_dir=args.plots_dir.resolve(),
    )
    for item in summary:
        print(
            f"{item['accuracy_rank']}. {item['activation']}: "
            f"val={item['mean_val_accuracy_pct']:.4f}% | "
            f"FLOPs={item['gflops_per_run']:.7f} GFLOPs | "
            f"{item['pareto_status']}"
        )
    return summary


if __name__ == "__main__":
    main()
