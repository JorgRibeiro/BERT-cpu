"""Regenerate the joint V1/V2/V3 performance-versus-FLOPs analysis.

Decisions in this module use only the frozen epoch-100 validation metric and
training FLOPs. Official-test accuracy is read from the already sealed
evaluation artifacts and is reported only as a post-freeze description.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments import evaluate_official_test as official
from experiments import plot_v1, plot_v2, plot_v3


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OFFICIAL_ARTIFACTS = REPO_ROOT / "experiments/final_evaluation"
DEFAULT_OUTPUT = REPO_ROOT / "experiments/final_analysis"
DEFAULT_RAW_PAIRS = DEFAULT_OUTPUT / "raw_pairs.csv"
DEFAULT_SUMMARY = DEFAULT_OUTPUT / "summary.csv"
DEFAULT_MARGINAL_RETURNS = DEFAULT_OUTPUT / "marginal_returns.csv"
DEFAULT_ANALYSIS = DEFAULT_OUTPUT / "analysis.md"
DEFAULT_PLOTS = DEFAULT_OUTPUT / "plots"

CONFIG_ORDER = official.EXPECTED_CONFIG_ORDER
SEEDS = official.EXPECTED_SEEDS
PRIMARY_METRIC = "validation_accuracy_epoch_100"
TEST_ROLE = "descriptive_only_after_freeze"
RELEVANT_GAIN_PP = 0.5
RELU_CONFIG = "F-RELU"
V2_REFERENCE = "S-BETA-1"

VARIABLE_BY_CONFIG = {
    **{config_id: "V1" for config_id in plot_v1.CONFIG_ORDER},
    **{config_id: "V2" for config_id in plot_v2.CONFIG_ORDER},
    **{config_id: "V3" for config_id in plot_v3.CONFIG_ORDER},
}

SHORT_LABELS = {
    "F-RELU": "ReLU",
    "F-SIGMOID": "Sigmoid",
    "F-SWISH": "Swish",
    "F-SOFTPLUS": "Softplus",
    "S-BETA-0.5": "β=0,5",
    "S-BETA-1": "β=1",
    "S-BETA-2": "β=2",
    "S-BETA-5": "β=5",
    "L1-DIRECT": "L1",
    "L2-IDENTITY": "L2",
    "L3-IDENTITY": "L3",
}

COLORS = {
    "F-RELU": "#1f77b4",
    "F-SIGMOID": "#4c9ed9",
    "F-SWISH": "#155a8a",
    "F-SOFTPLUS": "#7db7df",
    "S-BETA-0.5": "#ffbb78",
    "S-BETA-1": "#f28e2b",
    "S-BETA-2": "#d66b00",
    "S-BETA-5": "#9c4a00",
    "L1-DIRECT": "#59a14f",
    "L2-IDENTITY": "#2f7d32",
    "L3-IDENTITY": "#145a20",
}

RAW_PAIR_FIELDS = (
    "evaluation_id",
    "variable",
    "config_id",
    "source_run_id",
    "seed",
    "parameters",
    "train_loss_final",
    "validation_loss_final",
    "train_accuracy_pct",
    "validation_accuracy_pct",
    "test_accuracy_pct",
    "training_flops_total",
    "training_gflops",
    "validation_inference_flops_per_sample",
    "test_inference_flops_per_sample",
    "predictions_hash",
    "validation_majority_accuracy_pct",
    "test_majority_accuracy_pct",
)

SUMMARY_FIELDS = (
    "validation_rank",
    "test_rank_descriptive",
    "variable",
    "config_id",
    "parameters",
    "mean_validation_accuracy_pct",
    "std_validation_accuracy_pp",
    "mean_test_accuracy_pct",
    "std_test_accuracy_pp",
    "training_gflops_per_run",
    "test_inference_flops_per_sample",
    "return_pp_per_gflop",
    "within_relu_budget",
    "budget_choice",
    "pareto_validation",
    "dominated_by",
)

MARGINAL_FIELDS = (
    "scope",
    "from_config_id",
    "to_config_id",
    "delta_validation_accuracy_pp",
    "delta_training_gflops",
    "marginal_return_pp_per_gflop",
    "gain_relevant_0_5pp",
    "status",
)


@dataclass(frozen=True)
class JointDecisions:
    """Validation-only decisions plus one explicitly descriptive test result."""

    pareto_frontier: tuple[str, ...]
    best_return_config: str
    relu_budget_gflops: float
    budget_choice_config: str
    highest_validation_config: str
    highest_test_config_descriptive: str


def _canonical_index(config_id: str) -> int:
    return CONFIG_ORDER.index(config_id)


def _source_runs() -> dict[str, Any]:
    """Strictly load the 33 primary source runs without reading Adult test."""

    grouped_sets = (
        plot_v1.load_v1_runs(),
        plot_v2.load_v2_runs(),
        plot_v3.load_v3_runs(),
    )
    by_run_id: dict[str, Any] = {}
    for grouped in grouped_sets:
        for runs in grouped.values():
            for run in runs:
                run_id = run.row["run_id"]
                if run_id in by_run_id:
                    raise ValueError(f"duplicated source run: {run_id}")
                by_run_id[run_id] = run
    if tuple(
        f"{config_id}-s{seed}-r1"
        for config_id in CONFIG_ORDER
        for seed in SEEDS
    ) != tuple(
        f"{config_id}-s{seed}-r1"
        for config_id in CONFIG_ORDER
        for seed in SEEDS
        if f"{config_id}-s{seed}-r1" in by_run_id
    ):
        raise ValueError("source runs differ from the frozen 11x3 grid")
    if len(by_run_id) != 33:
        raise ValueError("joint analysis requires exactly 33 primary runs")
    return by_run_id


def _validate_source_link(source: Any, recorded: Mapping[str, str]) -> None:
    row = source.row
    exact_fields = {
        "source_run_id": row["run_id"],
        "variable": row["variable"],
        "config_id": row["config_id"],
        "activation": row["activation"],
        "seed": row["seed"],
        "checkpoint_path": row["checkpoint_path"],
        "checkpoint_hash": row["checkpoint_hash"],
        "final_weights_hash": row["final_weights_hash"],
        "source_commit": row["commit"],
        "val_accuracy": row["val_accuracy"],
        "parameters": row["parameters"],
        "training_flops_total": row["flops_total"],
        "status": "completed_valid",
    }
    if any(str(recorded[field]) != str(value) for field, value in exact_fields.items()):
        raise ValueError(f"official/source mismatch: {row['run_id']}")
    if (
        row["purpose"] != "primary"
        or row["run_kind"] != "scientific"
        or int(row["repetition"]) != 1
        or str(row["test_accuracy"]).strip()
    ):
        raise ValueError(f"invalid source role: {row['run_id']}")


def load_raw_pairs(
    official_artifacts: Path = DEFAULT_OFFICIAL_ARTIFACTS,
) -> tuple[list[dict[str, Any]], str]:
    """Join strict source runs to the sealed official CSV without test loading."""

    output, official_rows = official.verify_saved_artifacts(official_artifacts)
    if (
        output.get("training_performed") is not False
        or int(output.get("configuration_count", -1)) != 11
        or int(output.get("seed_count", -1)) != 3
        or int(output.get("checkpoint_count", -1)) != 33
    ):
        raise ValueError("official output violates the frozen no-training grid")
    sources = _source_runs()
    majority_values = {float(run.majority_accuracy) for run in sources.values()}
    if len(majority_values) != 1:
        raise ValueError("validation majority accuracy changed between runs")
    validation_majority_pct = majority_values.pop() * 100.0
    test_majority_pct = float(output["test_data"]["majority_accuracy"]) * 100.0
    evaluation_id = str(output["evaluation_id"])

    pairs: list[dict[str, Any]] = []
    for recorded in official_rows:
        run_id = recorded["source_run_id"]
        if run_id not in sources:
            raise ValueError(f"official result has no source run: {run_id}")
        source = sources[run_id]
        _validate_source_link(source, recorded)
        row = source.row
        config_id = row["config_id"]
        pairs.append(
            {
                "evaluation_id": evaluation_id,
                "variable": VARIABLE_BY_CONFIG[config_id],
                "config_id": config_id,
                "source_run_id": run_id,
                "seed": int(row["seed"]),
                "parameters": int(row["parameters"]),
                "train_loss_final": float(row["train_loss_final"]),
                "validation_loss_final": float(row["val_loss_final"]),
                "train_accuracy_pct": float(row["train_accuracy"]) * 100.0,
                "validation_accuracy_pct": float(row["val_accuracy"]) * 100.0,
                "test_accuracy_pct": float(recorded["test_accuracy"]) * 100.0,
                "training_flops_total": int(row["flops_total"]),
                "training_gflops": int(row["flops_total"]) / 1e9,
                "validation_inference_flops_per_sample": float(
                    row["inference_flops_per_sample"]
                ),
                "test_inference_flops_per_sample": int(
                    recorded["inference_flops_per_sample"]
                ),
                "predictions_hash": str(recorded["predictions_hash"]),
                "validation_majority_accuracy_pct": validation_majority_pct,
                "test_majority_accuracy_pct": test_majority_pct,
            }
        )

    expected_ids = [
        f"{config_id}-s{seed}-r1"
        for config_id in CONFIG_ORDER
        for seed in SEEDS
    ]
    if [pair["source_run_id"] for pair in pairs] != expected_ids:
        raise ValueError("joined rows differ from the frozen official order")
    for seed in SEEDS:
        standard_hash = next(
            pair["predictions_hash"]
            for pair in pairs
            if pair["config_id"] == "F-SOFTPLUS" and pair["seed"] == seed
        )
        beta_one_hash = next(
            pair["predictions_hash"]
            for pair in pairs
            if pair["config_id"] == "S-BETA-1" and pair["seed"] == seed
        )
        if standard_hash != beta_one_hash:
            raise ValueError(f"Softplus prediction identity failed for seed {seed}")
    return pairs, evaluation_id


def _dominates(candidate: Mapping[str, Any], other: Mapping[str, Any]) -> bool:
    """Return whether ``other`` dominates ``candidate`` on validation and cost."""

    candidate_cost = float(candidate["training_gflops_per_run"])
    other_cost = float(other["training_gflops_per_run"])
    candidate_accuracy = float(candidate["mean_validation_accuracy_pct"])
    other_accuracy = float(other["mean_validation_accuracy_pct"])
    return (
        other_cost <= candidate_cost
        and other_accuracy >= candidate_accuracy
        and (other_cost < candidate_cost or other_accuracy > candidate_accuracy)
    )


def aggregate_pairs(
    raw_pairs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], JointDecisions]:
    """Aggregate three seeds and make every selection from validation only."""

    if len(raw_pairs) != 33:
        raise ValueError("joint analysis requires exactly 33 raw pairs")
    if len({str(pair["evaluation_id"]) for pair in raw_pairs}) != 1:
        raise ValueError("raw pairs mix official evaluations")

    grouped: dict[str, list[Mapping[str, Any]]] = {
        config_id: [] for config_id in CONFIG_ORDER
    }
    for pair in raw_pairs:
        config_id = str(pair["config_id"])
        if config_id not in grouped:
            raise ValueError(f"unexpected configuration: {config_id}")
        grouped[config_id].append(pair)

    majority_values = {
        float(pair["validation_majority_accuracy_pct"]) for pair in raw_pairs
    }
    if len(majority_values) != 1:
        raise ValueError("validation majority differs between raw pairs")
    validation_majority_pct = majority_values.pop()

    summary: list[dict[str, Any]] = []
    for config_id in CONFIG_ORDER:
        pairs = sorted(grouped[config_id], key=lambda pair: int(pair["seed"]))
        if [int(pair["seed"]) for pair in pairs] != list(SEEDS):
            raise ValueError(f"{config_id} does not contain seeds 0, 1 and 2")
        for field in (
            "variable",
            "parameters",
            "training_flops_total",
            "training_gflops",
            "test_inference_flops_per_sample",
        ):
            if len({pair[field] for pair in pairs}) != 1:
                raise ValueError(f"{field} changed between seeds for {config_id}")

        validation = np.asarray(
            [float(pair["validation_accuracy_pct"]) for pair in pairs],
            dtype=float,
        )
        test = np.asarray(
            [float(pair["test_accuracy_pct"]) for pair in pairs],
            dtype=float,
        )
        if not np.isfinite(validation).all() or not np.isfinite(test).all():
            raise ValueError(f"non-finite accuracy in {config_id}")
        gflops = float(pairs[0]["training_gflops"])
        summary.append(
            {
                "variable": str(pairs[0]["variable"]),
                "config_id": config_id,
                "parameters": int(pairs[0]["parameters"]),
                "mean_validation_accuracy_pct": float(validation.mean()),
                "std_validation_accuracy_pp": float(validation.std(ddof=1)),
                "mean_test_accuracy_pct": float(test.mean()),
                "std_test_accuracy_pp": float(test.std(ddof=1)),
                "training_gflops_per_run": gflops,
                "test_inference_flops_per_sample": int(
                    pairs[0]["test_inference_flops_per_sample"]
                ),
                "return_pp_per_gflop": (
                    float(validation.mean()) - validation_majority_pct
                )
                / gflops,
            }
        )

    for candidate in summary:
        dominators = [
            other["config_id"]
            for other in summary
            if other is not candidate and _dominates(candidate, other)
        ]
        candidate["pareto_validation"] = "pareto" if not dominators else "dominated"
        candidate["dominated_by"] = "|".join(dominators)

    validation_order = sorted(
        summary,
        key=lambda row: (
            -float(row["mean_validation_accuracy_pct"]),
            float(row["training_gflops_per_run"]),
            int(row["parameters"]),
            _canonical_index(str(row["config_id"])),
        ),
    )
    for rank, row in enumerate(validation_order, start=1):
        row["validation_rank"] = rank

    test_order = sorted(
        summary,
        key=lambda row: (
            -float(row["mean_test_accuracy_pct"]),
            _canonical_index(str(row["config_id"])),
        ),
    )
    previous_accuracy: float | None = None
    previous_rank = 0
    for position, row in enumerate(test_order, start=1):
        accuracy = float(row["mean_test_accuracy_pct"])
        if previous_accuracy is None or accuracy != previous_accuracy:
            previous_rank = position
            previous_accuracy = accuracy
        row["test_rank_descriptive"] = previous_rank

    by_id = {row["config_id"]: row for row in summary}
    relu_budget = float(by_id[RELU_CONFIG]["training_gflops_per_run"])
    eligible = [
        row
        for row in summary
        if float(row["training_gflops_per_run"]) <= relu_budget
    ]
    budget_choice = min(
        eligible,
        key=lambda row: (
            -float(row["mean_validation_accuracy_pct"]),
            float(row["training_gflops_per_run"]),
            int(row["parameters"]),
            _canonical_index(str(row["config_id"])),
        ),
    )
    for row in summary:
        row["within_relu_budget"] = (
            "yes"
            if float(row["training_gflops_per_run"]) <= relu_budget
            else "no"
        )
        row["budget_choice"] = (
            "yes" if row["config_id"] == budget_choice["config_id"] else "no"
        )

    frontier = tuple(
        row["config_id"]
        for row in sorted(
            (
                row
                for row in summary
                if row["pareto_validation"] == "pareto"
            ),
            key=lambda row: float(row["training_gflops_per_run"]),
        )
    )
    best_return = min(
        summary,
        key=lambda row: (
            -float(row["return_pp_per_gflop"]),
            float(row["training_gflops_per_run"]),
            int(row["parameters"]),
            _canonical_index(str(row["config_id"])),
        ),
    )["config_id"]
    decisions = JointDecisions(
        pareto_frontier=frontier,
        best_return_config=str(best_return),
        relu_budget_gflops=relu_budget,
        budget_choice_config=str(budget_choice["config_id"]),
        highest_validation_config=str(validation_order[0]["config_id"]),
        highest_test_config_descriptive=str(test_order[0]["config_id"]),
    )
    return summary, decisions


def _marginal_row(
    scope: str,
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    delta_accuracy = float(second["mean_validation_accuracy_pct"]) - float(
        first["mean_validation_accuracy_pct"]
    )
    delta_gflops = float(second["training_gflops_per_run"]) - float(
        first["training_gflops_per_run"]
    )
    if delta_gflops < 0:
        raise ValueError(f"negative marginal cost in {scope}")

    if delta_gflops == 0:
        marginal: float | None = None
        status = "equal_cost_no_marginal"
    else:
        marginal = delta_accuracy / delta_gflops
        if delta_accuracy >= RELEVANT_GAIN_PP:
            status = "relevant_gain"
        elif delta_accuracy > 0:
            status = "positive_below_threshold"
        elif delta_accuracy < 0:
            status = "accuracy_decrease"
        else:
            status = "no_accuracy_gain"
    return {
        "scope": scope,
        "from_config_id": first["config_id"],
        "to_config_id": second["config_id"],
        "delta_validation_accuracy_pp": delta_accuracy,
        "delta_training_gflops": delta_gflops,
        "marginal_return_pp_per_gflop": marginal,
        "gain_relevant_0_5pp": "yes" if delta_accuracy >= RELEVANT_GAIN_PP else "no",
        "status": status,
    }


def calculate_marginal_returns(
    summary: Sequence[Mapping[str, Any]],
    decisions: JointDecisions,
) -> list[dict[str, Any]]:
    """Apply the frozen marginal-return rules within variables and globally."""

    by_id = {str(row["config_id"]): row for row in summary}
    rows: list[dict[str, Any]] = []

    frontier = [by_id[config_id] for config_id in decisions.pareto_frontier]
    rows.extend(
        _marginal_row("global_pareto", first, second)
        for first, second in zip(frontier, frontier[1:])
    )

    relu = by_id[RELU_CONFIG]
    rows.extend(
        _marginal_row("V1_vs_relu", relu, by_id[config_id])
        for config_id in plot_v1.CONFIG_ORDER
        if config_id != RELU_CONFIG
    )

    beta_one = by_id[V2_REFERENCE]
    rows.extend(
        _marginal_row("V2_vs_beta_1", beta_one, by_id[config_id])
        for config_id in plot_v2.CONFIG_ORDER
        if config_id != V2_REFERENCE
    )

    v3 = [by_id[config_id] for config_id in plot_v3.CONFIG_ORDER]
    rows.extend(
        _marginal_row("V3_adjacent_depth", first, second)
        for first, second in zip(v3, v3[1:])
    )
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.10f}"
    return value


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    """Write a deterministic, LF-terminated derived table."""

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
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                {field: _csv_value(row.get(field)) for field in fields}
                for row in rows
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _pt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _parameters(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def build_analysis_markdown(
    summary: Sequence[Mapping[str, Any]],
    decisions: JointDecisions,
    marginal_returns: Sequence[Mapping[str, Any]],
    *,
    evaluation_id: str,
) -> str:
    """Build the concise, presentation-ready joint interpretation."""

    by_id = {str(row["config_id"]): row for row in summary}
    ranked = sorted(summary, key=lambda row: int(row["validation_rank"]))
    global_marginals = [
        row for row in marginal_returns if row["scope"] == "global_pareto"
    ]
    v3_marginals = [
        row for row in marginal_returns if row["scope"] == "V3_adjacent_depth"
    ]
    dominated = [
        row for row in summary if row["pareto_validation"] == "dominated"
    ]

    l1 = by_id["L1-DIRECT"]
    l2 = by_id["L2-IDENTITY"]
    l3 = by_id["L3-IDENTITY"]
    relu = by_id[RELU_CONFIG]
    beta5 = by_id["S-BETA-5"]
    validation_amplitude_v2 = max(
        float(by_id[config_id]["mean_validation_accuracy_pct"])
        for config_id in plot_v2.CONFIG_ORDER
    ) - min(
        float(by_id[config_id]["mean_validation_accuracy_pct"])
        for config_id in plot_v2.CONFIG_ORDER
    )
    v3_cost_delta = float(l3["training_gflops_per_run"]) - float(
        l1["training_gflops_per_run"]
    )
    v3_accuracy_delta = float(l3["mean_validation_accuracy_pct"]) - float(
        l1["mean_validation_accuracy_pct"]
    )
    v3_cost_ratio = float(l3["training_gflops_per_run"]) / float(
        l1["training_gflops_per_run"]
    )
    relu_over_l1 = float(relu["mean_validation_accuracy_pct"]) - float(
        l1["mean_validation_accuracy_pct"]
    )
    next_best_return = max(
        float(row["return_pp_per_gflop"])
        for row in summary
        if row["config_id"] != decisions.best_return_config
    )
    return_ratio = float(l1["return_pp_per_gflop"]) / next_best_return
    first_summary = next(iter(summary))
    validation_majority_pct = float(
        first_summary["mean_validation_accuracy_pct"]
    ) - (
        float(first_summary["return_pp_per_gflop"])
        * float(first_summary["training_gflops_per_run"])
    )

    lines = [
        "# Análise conjunta — V1, V2 e V3",
        "",
        f"A avaliação de origem é `{evaluation_id}`. A tabela usa 33 pares brutos:",
        "11 configurações, seeds 0, 1 e 2.",
        "",
        "## Regra de leitura",
        "",
        "- Decisões, retorno, orçamento e Pareto usam somente a média de validação",
        "  da época 100 e os FLOPs instrumentados de treinamento.",
        "- Retorno: `(validação − classe majoritária) / GFLOPs de treinamento`;",
        f"  maioria da validação = {_pt(validation_majority_pct, 4)}%.",
        f"- Ganho relevante: `{_pt(RELEVANT_GAIN_PP, 1)}` p.p.; orçamento:",
        f"  `{_pt(decisions.relu_budget_gflops, 7)}` GFLOPs de `F-RELU`.",
        "- O teste oficial aparece somente como descrição pós-congelamento; ele",
        "  não muda hipóteses, seleção, Pareto ou novas execuções.",
        "",
        "## Tabela conjunta",
        "",
        "| Rank val | Configuração | Variável | Parâmetros | Validação ± DP | Teste ± DP* | GFLOPs | Retorno | Estado |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        lines.append(
            "| "
            f"{row['validation_rank']} | `{row['config_id']}` | {row['variable']} | "
            f"{_parameters(int(row['parameters']))} | "
            f"{_pt(float(row['mean_validation_accuracy_pct']))}% ± "
            f"{_pt(float(row['std_validation_accuracy_pp']))} | "
            f"{_pt(float(row['mean_test_accuracy_pct']))}% ± "
            f"{_pt(float(row['std_test_accuracy_pp']))} | "
            f"{_pt(float(row['training_gflops_per_run']), 7)} | "
            f"{_pt(float(row['return_pp_per_gflop']), 6)} | "
            f"{row['pareto_validation']} |"
        )
    lines.extend(
        [
            "",
            "\\* Teste apenas descritivo.",
            "",
            "## Pareto global por validação",
            "",
            " → ".join(f"`{config_id}`" for config_id in decisions.pareto_frontier),
            "",
            "Retornos marginais entre vizinhos:",
            "",
            "| Transição | Δ validação | Δ GFLOPs | Retorno marginal |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in global_marginals:
        lines.append(
            f"| `{row['from_config_id']}` → `{row['to_config_id']}` | "
            f"{_pt(float(row['delta_validation_accuracy_pp']), 6)} p.p. | "
            f"{_pt(float(row['delta_training_gflops']), 6)} | "
            f"{_pt(float(row['marginal_return_pp_per_gflop']), 6)} p.p./GFLOP |"
        )
    lines.extend(
        [
            "",
            "Dominadas, com ao menos uma testemunha registrada no CSV:",
            "",
            *[
                f"- `{row['config_id']}` por `{str(row['dominated_by']).split('|')[0]}`."
                for row in dominated
            ],
            "",
            "## Quatro respostas obrigatórias",
            "",
            "1. **Melhor retorno por FLOP:** `L1-DIRECT`, com",
            f"   `{_pt(float(l1['return_pp_per_gflop']), 6)}` p.p./GFLOP, cerca de",
            f"   `{_pt(return_ratio, 1)}` vezes o segundo melhor retorno. Ele é",
            "   barato, mas não tem a maior acurácia.",
            "2. **Retornos decrescentes:** pelo limiar congelado, eles aparecem já",
            "   em `L1 → L2`: o ganho é de apenas",
            f"   `{_pt(float(l2['mean_validation_accuracy_pct']) - float(l1['mean_validation_accuracy_pct']), 6)}`",
            "   p.p. para",
            f"   `{_pt(float(l2['training_gflops_per_run']) - float(l1['training_gflops_per_run']), 6)}`",
            "   GFLOPs. Em V3, `L2 → L3` cai para",
            f"   `{_pt(float(v3_marginals[1]['marginal_return_pp_per_gflop']), 7)}`",
            "   p.p./GFLOP. A fronteira global não tem inclinações monotonicamente",
            "   decrescentes; portanto não há um único cotovelo suave.",
            "3. **Maior mudança de custo com pouco desempenho:** V3. De L1 para L3,",
            f"   o custo cresce `{_pt(v3_cost_delta, 6)}` GFLOPs",
            f"   (`{_pt(v3_cost_ratio, 2)}x`) para somente",
            f"   `{_pt(v3_accuracy_delta, 6)}` p.p. Em sentido contrário, V2 muda",
            f"   `{_pt(validation_amplitude_v2, 6)}` p.p. entre betas com o mesmo",
            "   custo instrumentado; ainda assim, o efeito fica abaixo de 0,5 p.p.",
            "4. **Escolha sob orçamento fixo:** `F-RELU`. Entre L1, L2 e ReLU,",
            f"   ela tem a maior validação (`{_pt(float(relu['mean_validation_accuracy_pct']))}%`)",
            f"   e ganha `{_pt(relu_over_l1, 6)}` p.p. sobre L1, atingindo o limiar",
            "   relevante. Se o objetivo fosse eficiência absoluta, L1 seria a",
            "   escolha; sob o orçamento e priorizando acurácia, é ReLU.",
            "",
            "## Leitura do teste oficial",
            "",
            f"- Maior validação: `{decisions.highest_validation_config}` = "
            f"{_pt(float(beta5['mean_validation_accuracy_pct']))}%.",
            f"- Maior teste descritivo: `{decisions.highest_test_config_descriptive}` = "
            f"{_pt(float(by_id[decisions.highest_test_config_descriptive]['mean_test_accuracy_pct']))}%.",
            "- A troca de ordem é uma observação pós-congelamento, não motivo para",
            "  escolher outra configuração ou repetir treinamento.",
            "",
            "## Limitações",
            "",
            "- Três seeds no mesmo split medem variação de inicialização, não de",
            "  amostragem; DP não é intervalo de confiança.",
            "- FLOPs são instrumentados e não medem tempo, energia ou custo completo.",
            "- O encoder foi ajustado antes do hold-out, sem rótulos da validação.",
            "- `F-SOFTPLUS` e `S-BETA-1` permanecem IDs distintos; têm métricas e",
            "  predições iguais, mas instrumentações de custo diferentes.",
            "- V3 mantém o risco acadêmico de ser interpretada como variável",
            "  arquitetural, não como terceira variável de q01.",
            "- A suíte ampla de desenvolvimento foi executada depois da avaliação",
            "  oficial e inclui um teste do loader que acessa `adult.test`, além de",
            "  treinos curtos. Esse acesso não executou os checkpoints oficiais,",
            "  não gerou métricas de modelos e não alterou decisões ou resultados.",
            "",
            "## Reprodução",
            "",
            "```bash",
            "python -m experiments.evaluate_official_test --verify-only",
            "pytest -q test/test_plot_joint.py",
            "python -m experiments.plot_joint",
            "python -m experiments.evaluate_official_test --verify-only",
            "```",
            "",
            "Resultado esperado do teste focado: `6 passed`. O gerador apenas",
            "verifica e lê a avaliação oficial já salva; não carrega o Adult test e",
            "não executa treinamento. O comando amplo de pytest não faz parte desta",
            "rota segura.",
            "",
        ]
    )
    return "\n".join(lines)


def _prepare_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.22,
            "font.size": 10,
            "figure.dpi": 110,
        }
    )
    return plt


def _pairs_by_config(
    raw_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    return {
        config_id: [
            pair for pair in raw_pairs if pair["config_id"] == config_id
        ]
        for config_id in CONFIG_ORDER
    }


def plot_accuracy_vs_training_flops(
    path: Path,
    raw_pairs: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
) -> None:
    """Plot raw validation points, means and the validation Pareto frontier."""

    plt = _prepare_matplotlib()
    grouped = _pairs_by_config(raw_pairs)
    by_id = {str(row["config_id"]): row for row in summary}
    figure, (full, zoom) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for axis in (full, zoom):
        for config_id in CONFIG_ORDER:
            row = by_id[config_id]
            x = float(row["training_gflops_per_run"])
            values = [
                float(pair["validation_accuracy_pct"])
                for pair in grouped[config_id]
            ]
            axis.scatter(
                [x] * len(values),
                values,
                s=24,
                facecolor="none",
                edgecolor=COLORS[config_id],
                alpha=0.75,
                linewidth=0.9,
                zorder=2,
            )
            axis.errorbar(
                x,
                float(row["mean_validation_accuracy_pct"]),
                yerr=float(row["std_validation_accuracy_pp"]),
                fmt="D",
                markersize=6.5,
                capsize=3,
                color=COLORS[config_id],
                markeredgecolor="black",
                markeredgewidth=0.45,
                zorder=3,
            )

        frontier = sorted(
            (
                row
                for row in summary
                if row["pareto_validation"] == "pareto"
            ),
            key=lambda row: float(row["training_gflops_per_run"]),
        )
        axis.plot(
            [float(row["training_gflops_per_run"]) for row in frontier],
            [float(row["mean_validation_accuracy_pct"]) for row in frontier],
            color="#555555",
            linestyle="--",
            linewidth=1.3,
            label="Pareto por validação",
            zorder=1,
        )

    full.set_xlim(0, 160)
    full.set_ylim(84.45, 85.40)
    full.set_title("Visão completa")
    for config_id in ("L1-DIRECT", "L2-IDENTITY", "F-RELU", "S-BETA-5", "L3-IDENTITY"):
        row = by_id[config_id]
        full.annotate(
            SHORT_LABELS[config_id],
            (
                float(row["training_gflops_per_run"]),
                float(row["mean_validation_accuracy_pct"]),
            ),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8.5,
        )

    zoom.set_xlim(83.85, 85.22)
    zoom.set_title("Zoom das configurações próximas de 84–85 GFLOPs")
    offsets = {
        "L2-IDENTITY": (-8, -20),
        "F-RELU": (-12, 10),
        "F-SOFTPLUS": (-22, 10),
        "F-SIGMOID": (-20, -20),
        "F-SWISH": (-45, -20),
        "S-BETA-0.5": (8, -27),
        "S-BETA-1": (8, -13),
        "S-BETA-2": (8, 1),
        "S-BETA-5": (8, 15),
    }
    for config_id, offset in offsets.items():
        row = by_id[config_id]
        zoom.annotate(
            SHORT_LABELS[config_id],
            (
                float(row["training_gflops_per_run"]),
                float(row["mean_validation_accuracy_pct"]),
            ),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.2,
        )
    full.set_ylabel("Acurácia de validação na época 100 (%)")
    for axis in (full, zoom):
        axis.set_xlabel("GFLOPs instrumentados de treinamento por run")
    full.legend(loc="lower left")
    figure.suptitle(
        "Análise conjunta — desempenho versus custo\n"
        "círculos = seeds; losangos = média ± DP",
        fontsize=13,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_validation_vs_test(
    path: Path,
    raw_pairs: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
) -> None:
    """Compare validation with the frozen, descriptive official-test result."""

    plt = _prepare_matplotlib()
    grouped = _pairs_by_config(raw_pairs)
    figure, axis = plt.subplots(figsize=(8.5, 7))
    annotation_offsets = {
        "F-RELU": (6, 6),
        "F-SIGMOID": (6, 4),
        "F-SWISH": (6, 4),
        "F-SOFTPLUS": (6, 10),
        "S-BETA-0.5": (8, 7),
        "S-BETA-1": (8, -13),
        "S-BETA-2": (8, 6),
        "S-BETA-5": (8, 6),
        "L1-DIRECT": (6, 5),
        "L2-IDENTITY": (-25, 9),
        "L3-IDENTITY": (7, -13),
    }
    for row in summary:
        config_id = str(row["config_id"])
        pairs = grouped[config_id]
        axis.scatter(
            [float(pair["validation_accuracy_pct"]) for pair in pairs],
            [float(pair["test_accuracy_pct"]) for pair in pairs],
            s=25,
            facecolor="none",
            edgecolor=COLORS[config_id],
            alpha=0.7,
        )
        axis.errorbar(
            float(row["mean_validation_accuracy_pct"]),
            float(row["mean_test_accuracy_pct"]),
            xerr=float(row["std_validation_accuracy_pp"]),
            yerr=float(row["std_test_accuracy_pp"]),
            fmt="D",
            markersize=7,
            capsize=3,
            color=COLORS[config_id],
            markeredgecolor="black",
            markeredgewidth=0.45,
        )
        axis.annotate(
            SHORT_LABELS[config_id],
            (
                float(row["mean_validation_accuracy_pct"]),
                float(row["mean_test_accuracy_pct"]),
            ),
            xytext=annotation_offsets[config_id],
            textcoords="offset points",
            fontsize=8,
        )
    lower, upper = 84.4, 85.85
    axis.plot(
        [lower, upper],
        [lower, upper],
        color="#777777",
        linestyle=":",
        linewidth=1,
        label="validação = teste",
    )
    axis.set(
        xlim=(lower, upper),
        ylim=(lower, upper),
        xlabel="Acurácia média de validação (%)",
        ylabel="Acurácia média no teste oficial (%)",
        title="Teste oficial descritivo — não usado para seleção",
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_return_per_flop(
    path: Path,
    summary: Sequence[Mapping[str, Any]],
) -> None:
    """Show validation return per training GFLOP on a readable log scale."""

    plt = _prepare_matplotlib()
    ordered = sorted(
        summary,
        key=lambda row: float(row["return_pp_per_gflop"]),
    )
    figure, axis = plt.subplots(figsize=(9, 6.5))
    for index, row in enumerate(ordered):
        config_id = str(row["config_id"])
        value = float(row["return_pp_per_gflop"])
        axis.hlines(index, 0.045, value, color=COLORS[config_id], alpha=0.45)
        axis.scatter(
            value,
            index,
            color=COLORS[config_id],
            edgecolor="black",
            linewidth=0.4,
            s=65 if config_id == "L1-DIRECT" else 45,
            zorder=3,
        )
        axis.annotate(
            _pt(value, 4),
            (value, index),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    axis.set_xscale("log")
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels([str(row["config_id"]) for row in ordered])
    axis.set(
        xlim=(0.045, 4.5),
        xlabel="Retorno de validação (p.p./GFLOP, escala log)",
        title="Retorno por FLOP instrumentado de treinamento",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def generate(
    *,
    official_artifacts: Path = DEFAULT_OFFICIAL_ARTIFACTS,
    output_dir: Path = DEFAULT_OUTPUT,
) -> tuple[list[dict[str, Any]], JointDecisions, list[dict[str, Any]]]:
    """Regenerate all joint tables, plots and the written interpretation."""

    raw_pairs, evaluation_id = load_raw_pairs(official_artifacts)
    summary, decisions = aggregate_pairs(raw_pairs)
    marginal_returns = calculate_marginal_returns(summary, decisions)

    write_csv(output_dir / "raw_pairs.csv", raw_pairs, RAW_PAIR_FIELDS)
    write_csv(output_dir / "summary.csv", summary, SUMMARY_FIELDS)
    write_csv(
        output_dir / "marginal_returns.csv",
        marginal_returns,
        MARGINAL_FIELDS,
    )
    _atomic_write_text(
        output_dir / "analysis.md",
        build_analysis_markdown(
            summary,
            decisions,
            marginal_returns,
            evaluation_id=evaluation_id,
        ),
    )
    plots_dir = output_dir / "plots"
    plot_accuracy_vs_training_flops(
        plots_dir / "accuracy_vs_training_flops.png",
        raw_pairs,
        summary,
    )
    plot_validation_vs_test(
        plots_dir / "validation_vs_test.png",
        raw_pairs,
        summary,
    )
    plot_return_per_flop(
        plots_dir / "return_per_flop.png",
        summary,
    )
    return summary, decisions, marginal_returns


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate the frozen V1/V2/V3 joint analysis.",
    )
    parser.add_argument(
        "--official-artifacts",
        type=Path,
        default=DEFAULT_OFFICIAL_ARTIFACTS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(
    argv: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], JointDecisions, list[dict[str, Any]]]:
    args = _build_parser().parse_args(argv)
    summary, decisions, marginal_returns = generate(
        official_artifacts=args.official_artifacts,
        output_dir=args.output_dir,
    )
    print(
        "Joint analysis: "
        f"{len(summary)} configurations; "
        f"Pareto={' -> '.join(decisions.pareto_frontier)}; "
        f"best return={decisions.best_return_config}; "
        f"ReLU-budget choice={decisions.budget_choice_config}; "
        "official test not loaded"
    )
    return summary, decisions, marginal_returns


if __name__ == "__main__":
    main()
