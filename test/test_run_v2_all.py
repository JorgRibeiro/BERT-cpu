"""Tests for the sequential and resumable Variable 2 batch."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import run_v2_all


def _run_id(config_id: str, seed: int, repetition: int = 1, smoke: bool = False) -> str:
    prefix = "SMOKE-" if smoke else ""
    return f"{prefix}{config_id}-s{seed}-r{repetition}"


def _valid_row(config_id: str, seed: int) -> dict[str, str]:
    return {
        "run_id": _run_id(config_id, seed),
        "config_id": config_id,
        "seed": str(seed),
        "run_kind": "scientific",
        "status": "completed_valid",
    }


@pytest.fixture
def public_runner_api(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(run_v2_all.run_v2, "load_config", lambda path: {})
    monkeypatch.setattr(
        run_v2_all.run_v2,
        "validate_frozen_config",
        lambda config: None,
    )
    monkeypatch.setattr(run_v2_all.run_v2, "expected_run_id", _run_id)
    monkeypatch.setattr(
        run_v2_all.run_v2,
        "result_artifacts_valid",
        lambda row, *, config_path: row.get("artifacts_valid", "yes") == "yes",
    )


def test_run_plan_has_reference_first_and_exactly_twelve_unique_runs():
    assert run_v2_all.CONFIG_ORDER == (
        "S-BETA-1",
        "S-BETA-0.5",
        "S-BETA-2",
        "S-BETA-5",
    )
    assert run_v2_all.RUN_PLAN == (
        ("S-BETA-1", 0),
        ("S-BETA-1", 1),
        ("S-BETA-1", 2),
        ("S-BETA-0.5", 0),
        ("S-BETA-0.5", 1),
        ("S-BETA-0.5", 2),
        ("S-BETA-2", 0),
        ("S-BETA-2", 1),
        ("S-BETA-2", 2),
        ("S-BETA-5", 0),
        ("S-BETA-5", 1),
        ("S-BETA-5", 2),
    )
    assert len(set(run_v2_all.RUN_PLAN)) == 12


def test_dry_run_skips_only_valid_complete_row_and_never_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_runner_api,
):
    monkeypatch.setattr(
        run_v2_all.run_v2,
        "load_results",
        lambda path: [_valid_row("S-BETA-1", 0)],
    )
    executed = []
    monkeypatch.setattr(
        run_v2_all.run_v2,
        "execute_run",
        lambda **kwargs: executed.append(kwargs),
    )

    outcomes = run_v2_all.execute_all(
        dry_run=True,
        config_path=tmp_path / "config.json",
        artifacts_dir=tmp_path,
    )

    assert outcomes[0]["action"] == "skip_valid"
    assert [item["action"] for item in outcomes[1:]] == ["would_run"] * 11
    assert executed == []


@pytest.mark.parametrize(
    "changed",
    [
        {"status": "failed"},
        {"status": "running"},
        {"run_kind": "smoke"},
        {"artifacts_valid": "no"},
    ],
)
def test_registered_incomplete_or_invalid_run_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_runner_api,
    changed: dict[str, str],
):
    row = {**_valid_row("S-BETA-1", 0), **changed}
    monkeypatch.setattr(run_v2_all.run_v2, "load_results", lambda path: [row])
    executed = []
    monkeypatch.setattr(
        run_v2_all.run_v2,
        "execute_run",
        lambda **kwargs: executed.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        run_v2_all.execute_all(
            config_path=tmp_path / "config.json",
            artifacts_dir=tmp_path,
        )
    assert executed == []


def test_duplicate_or_unregistered_partial_artifact_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_runner_api,
):
    row = _valid_row("S-BETA-1", 0)
    monkeypatch.setattr(run_v2_all.run_v2, "load_results", lambda path: [row, row])
    with pytest.raises(RuntimeError, match="duplicate result"):
        run_v2_all.execute_all(
            dry_run=True,
            config_path=tmp_path / "config.json",
            artifacts_dir=tmp_path,
        )

    monkeypatch.setattr(run_v2_all.run_v2, "load_results", lambda path: [])
    partial = tmp_path / "logs" / "S-BETA-1-s0-r1.jsonl"
    partial.parent.mkdir(parents=True)
    partial.write_text('{"status":"running"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="unregistered artifacts"):
        run_v2_all.execute_all(
            dry_run=True,
            config_path=tmp_path / "config.json",
            artifacts_dir=tmp_path,
        )


def test_unexpected_registered_run_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_runner_api,
):
    unexpected = _valid_row("S-BETA-99", 0)
    monkeypatch.setattr(
        run_v2_all.run_v2,
        "load_results",
        lambda path: [unexpected],
    )

    with pytest.raises(RuntimeError, match="unexpected run IDs"):
        run_v2_all.execute_all(
            dry_run=True,
            config_path=tmp_path / "config.json",
            artifacts_dir=tmp_path,
        )


def test_executes_strictly_in_order_without_test_or_parallel_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_runner_api,
):
    monkeypatch.setattr(run_v2_all.run_v2, "load_results", lambda path: [])
    calls: list[dict] = []

    def fake_execute(**kwargs):
        calls.append(kwargs)
        return _valid_row(kwargs["config_id"], kwargs["seed"])

    monkeypatch.setattr(run_v2_all.run_v2, "execute_run", fake_execute)
    outcomes = run_v2_all.execute_all(
        config_path=tmp_path / "config.json",
        artifacts_dir=tmp_path,
        verbose=False,
    )

    assert [(call["config_id"], call["seed"]) for call in calls] == list(
        run_v2_all.RUN_PLAN
    )
    assert [item["action"] for item in outcomes] == ["executed"] * 12
    assert all(call["smoke"] is False for call in calls)
    assert all(call["repetition"] == 1 for call in calls)
    assert all("evaluate_test" not in call for call in calls)
    assert all("--evaluate-test" not in call["command"] for call in calls)
    assert all("--quiet" in call["command"] for call in calls)


def test_failure_aborts_before_later_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_runner_api,
):
    monkeypatch.setattr(run_v2_all.run_v2, "load_results", lambda path: [])
    calls = []

    def fail_second(**kwargs):
        calls.append((kwargs["config_id"], kwargs["seed"]))
        if len(calls) == 2:
            raise RuntimeError("training failed")
        return _valid_row(kwargs["config_id"], kwargs["seed"])

    monkeypatch.setattr(run_v2_all.run_v2, "execute_run", fail_second)
    with pytest.raises(RuntimeError, match="training failed"):
        run_v2_all.execute_all(
            config_path=tmp_path / "config.json",
            artifacts_dir=tmp_path,
        )

    assert calls == [("S-BETA-1", 0), ("S-BETA-1", 1)]


def test_cli_dry_run_forwards_paths_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    received = {}

    def fake_execute_all(**kwargs):
        received.update(kwargs)
        return [
            {
                "run_id": "S-BETA-1-s0-r1",
                "config_id": "S-BETA-1",
                "seed": 0,
                "action": "would_run",
            }
        ]

    monkeypatch.setattr(run_v2_all, "execute_all", fake_execute_all)
    config_path = tmp_path / "v2.json"
    artifacts_dir = tmp_path / "artifacts"

    outcomes = run_v2_all.main(
        [
            "--dry-run",
            "--quiet",
            "--config",
            str(config_path),
            "--artifacts-dir",
            str(artifacts_dir),
        ]
    )

    assert outcomes[0]["action"] == "would_run"
    assert received == {
        "dry_run": True,
        "config_path": config_path,
        "artifacts_dir": artifacts_dir,
        "verbose": False,
    }
