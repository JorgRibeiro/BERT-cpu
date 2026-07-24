"""Regression tests for the frozen joint V1/V2/V3 analysis."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from experiments import plot_joint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sealed_hashes() -> dict[str, str]:
    root = plot_joint.DEFAULT_OFFICIAL_ARTIFACTS
    return {
        name: _sha256(root / name)
        for name in (
            "input_manifest.json",
            "evaluation.jsonl",
            "results.csv",
            "output_manifest.json",
        )
    }


def test_joint_loader_uses_saved_results_without_loading_adult_test(monkeypatch):
    before = _sealed_hashes()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("joint analysis must not load or reevaluate Adult test")

    monkeypatch.setattr(plot_joint.official.datasets, "load_adult", forbidden)
    monkeypatch.setattr(plot_joint.official.adult, "train", forbidden)
    monkeypatch.setattr(plot_joint.official, "_forward_once", forbidden)
    monkeypatch.setattr(plot_joint.official, "run_official_evaluation", forbidden)
    monkeypatch.setattr(plot_joint.official, "build_input_manifest", forbidden)
    monkeypatch.setattr(plot_joint.official, "prepare_input_manifest", forbidden)

    pairs, evaluation_id = plot_joint.load_raw_pairs()

    assert evaluation_id == "OFFICIAL-185889b9b944304ba514"
    assert len(pairs) == 33
    assert [pair["config_id"] for pair in pairs[::3]] == list(
        plot_joint.CONFIG_ORDER
    )
    assert all(
        [pair["seed"] for pair in pairs if pair["config_id"] == config_id]
        == [0, 1, 2]
        for config_id in plot_joint.CONFIG_ORDER
    )
    assert before == _sealed_hashes()

    monkeypatch.setattr(
        plot_joint.official,
        "verify_saved_artifacts",
        lambda *_args, **_kwargs: (
            {
                "training_performed": True,
                "configuration_count": 11,
                "seed_count": 3,
                "checkpoint_count": 33,
            },
            [],
        ),
    )
    with pytest.raises(ValueError, match="no-training grid"):
        plot_joint.load_raw_pairs()


def test_joint_aggregate_reproduces_pareto_return_and_budget_choice():
    pairs, _ = plot_joint.load_raw_pairs()
    summary, decisions = plot_joint.aggregate_pairs(pairs)
    by_id = {row["config_id"]: row for row in summary}

    assert decisions.pareto_frontier == (
        "L1-DIRECT",
        "L2-IDENTITY",
        "F-RELU",
        "S-BETA-5",
    )
    assert decisions.best_return_config == "L1-DIRECT"
    assert decisions.relu_budget_gflops == 84.2375505
    assert decisions.budget_choice_config == "F-RELU"
    assert decisions.highest_validation_config == "S-BETA-5"
    assert decisions.highest_test_config_descriptive == "S-BETA-2"

    assert np.isclose(
        by_id["L1-DIRECT"]["return_pp_per_gflop"],
        3.4546571574378238,
    )
    assert np.isclose(
        by_id["S-BETA-5"]["mean_validation_accuracy_pct"],
        85.1965601965602,
    )
    assert np.isclose(
        by_id["S-BETA-2"]["mean_test_accuracy_pct"],
        85.64584484982495,
    )
    assert {
        row["config_id"]
        for row in summary
        if row["within_relu_budget"] == "yes"
    } == {"L1-DIRECT", "L2-IDENTITY", "F-RELU"}


def test_equal_cost_dominance_and_duplicate_experimental_ids_are_preserved():
    pairs, _ = plot_joint.load_raw_pairs()
    summary, _ = plot_joint.aggregate_pairs(pairs)
    by_id = {row["config_id"]: row for row in summary}

    assert by_id["F-SWISH"]["training_gflops_per_run"] == by_id[
        "S-BETA-5"
    ]["training_gflops_per_run"]
    assert by_id["F-SWISH"]["pareto_validation"] == "dominated"
    assert "S-BETA-5" in by_id["F-SWISH"]["dominated_by"].split("|")

    standard = by_id["F-SOFTPLUS"]
    beta_one = by_id["S-BETA-1"]
    assert standard["mean_validation_accuracy_pct"] == beta_one[
        "mean_validation_accuracy_pct"
    ]
    assert standard["mean_test_accuracy_pct"] == beta_one[
        "mean_test_accuracy_pct"
    ]
    assert standard["test_rank_descriptive"] == beta_one[
        "test_rank_descriptive"
    ]
    assert standard["training_gflops_per_run"] < beta_one[
        "training_gflops_per_run"
    ]
    assert "F-SOFTPLUS" in beta_one["dominated_by"].split("|")
    for seed in plot_joint.SEEDS:
        standard_prediction = next(
            pair["predictions_hash"]
            for pair in pairs
            if pair["config_id"] == "F-SOFTPLUS" and pair["seed"] == seed
        )
        beta_one_prediction = next(
            pair["predictions_hash"]
            for pair in pairs
            if pair["config_id"] == "S-BETA-1" and pair["seed"] == seed
        )
        assert standard_prediction == beta_one_prediction


def test_test_ranking_cannot_change_validation_only_decisions():
    pairs, _ = plot_joint.load_raw_pairs()
    original_summary, original = plot_joint.aggregate_pairs(pairs)

    changed_pairs = [dict(pair) for pair in pairs]
    for pair in changed_pairs:
        pair["test_accuracy_pct"] = float(
            60 + plot_joint._canonical_index(str(pair["config_id"]))
        )
    changed_summary, changed = plot_joint.aggregate_pairs(changed_pairs)

    assert changed.pareto_frontier == original.pareto_frontier
    assert changed.best_return_config == original.best_return_config
    assert changed.relu_budget_gflops == original.relu_budget_gflops
    assert changed.budget_choice_config == original.budget_choice_config
    assert changed.highest_validation_config == original.highest_validation_config
    assert changed.highest_test_config_descriptive != (
        original.highest_test_config_descriptive
    )
    assert [
        (
            row["config_id"],
            row["pareto_validation"],
            row["return_pp_per_gflop"],
            row["within_relu_budget"],
            row["budget_choice"],
        )
        for row in changed_summary
    ] == [
        (
            row["config_id"],
            row["pareto_validation"],
            row["return_pp_per_gflop"],
            row["within_relu_budget"],
            row["budget_choice"],
        )
        for row in original_summary
    ]


def test_marginal_returns_apply_equal_cost_and_relevance_rules():
    pairs, _ = plot_joint.load_raw_pairs()
    summary, decisions = plot_joint.aggregate_pairs(pairs)
    marginal = plot_joint.calculate_marginal_returns(summary, decisions)

    global_rows = [row for row in marginal if row["scope"] == "global_pareto"]
    assert [
        (row["from_config_id"], row["to_config_id"])
        for row in global_rows
    ] == [
        ("L1-DIRECT", "L2-IDENTITY"),
        ("L2-IDENTITY", "F-RELU"),
        ("F-RELU", "S-BETA-5"),
    ]
    assert all(row["gain_relevant_0_5pp"] == "no" for row in global_rows)
    assert np.isclose(
        global_rows[0]["marginal_return_pp_per_gflop"],
        0.0016346135117062679,
    )
    assert np.isclose(
        global_rows[1]["marginal_return_pp_per_gflop"],
        1.793120622010922,
    )

    v2_rows = [row for row in marginal if row["scope"] == "V2_vs_beta_1"]
    assert len(v2_rows) == 3
    assert all(row["delta_training_gflops"] == 0 for row in v2_rows)
    assert all(row["marginal_return_pp_per_gflop"] is None for row in v2_rows)
    assert all(row["status"] == "equal_cost_no_marginal" for row in v2_rows)

    v3_rows = [row for row in marginal if row["scope"] == "V3_adjacent_depth"]
    assert np.isclose(
        v3_rows[1]["marginal_return_pp_per_gflop"],
        0.00007267213057485274,
        rtol=1e-12,
        atol=1e-15,
    )
    assert v3_rows[1]["marginal_return_pp_per_gflop"] < v3_rows[0][
        "marginal_return_pp_per_gflop"
    ]


def test_joint_generator_writes_deterministic_tables_analysis_and_plots(tmp_path):
    before = _sealed_hashes()

    summary, decisions, marginal = plot_joint.generate(output_dir=tmp_path)
    first_text = {
        name: (tmp_path / name).read_bytes()
        for name in (
            "raw_pairs.csv",
            "summary.csv",
            "marginal_returns.csv",
            "analysis.md",
        )
    }
    plot_joint.generate(output_dir=tmp_path)

    assert len(summary) == 11
    assert decisions.budget_choice_config == "F-RELU"
    assert len(marginal) == 11
    assert before == _sealed_hashes()
    assert first_text == {
        name: (tmp_path / name).read_bytes()
        for name in first_text
    }

    with (tmp_path / "raw_pairs.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 33
    with (tmp_path / "summary.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 11
    analysis = (tmp_path / "analysis.md").read_text(encoding="utf-8")
    assert "O teste oficial aparece somente como descrição" in analysis
    assert "L1-DIRECT` → `L2-IDENTITY` → `F-RELU` → `S-BETA-5" in analysis

    expected_plots = {
        "accuracy_vs_training_flops.png",
        "validation_vs_test.png",
        "return_per_flop.png",
    }
    assert {path.name for path in (tmp_path / "plots").iterdir()} == expected_plots
    for name in expected_plots:
        data = (tmp_path / "plots" / name).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 20_000
