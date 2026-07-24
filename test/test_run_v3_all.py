"""Tests for the sequential and resumable Variable 3 batch."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments import run_v3_all


def _run_id(config_id: str, seed: int, repetition: int = 1, smoke: bool = False):
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
    monkeypatch.setattr(run_v3_all.run_v3, "load_config", lambda path: {})
    monkeypatch.setattr(
        run_v3_all.run_v3,
        "validate_frozen_config",
        lambda config: None,
    )
    monkeypatch.setattr(run_v3_all.run_v3, "expected_run_id", _run_id)
    monkeypatch.setattr(
        run_v3_all.run_v3,
        "result_artifacts_valid",
        lambda row, *, config_path: row.get("artifacts_valid", "yes") == "yes",
    )


def test_run_plan_has_direct_reference_first_and_nine_unique_runs():
    assert run_v3_all.CONFIG_ORDER == (
        "L1-DIRECT",
        "L2-IDENTITY",
        "L3-IDENTITY",
    )
    assert run_v3_all.RUN_PLAN == (
        ("L1-DIRECT", 0),
        ("L1-DIRECT", 1),
        ("L1-DIRECT", 2),
        ("L2-IDENTITY", 0),
        ("L2-IDENTITY", 1),
        ("L2-IDENTITY", 2),
        ("L3-IDENTITY", 0),
        ("L3-IDENTITY", 1),
        ("L3-IDENTITY", 2),
    )
    assert len(set(run_v3_all.RUN_PLAN)) == 9


def test_dry_run_skips_only_valid_complete_row_and_never_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_runner_api,
):
    monkeypatch.setattr(
        run_v3_all.run_v3,
        "load_results",
        lambda path: [_valid_row("L1-DIRECT", 0)],
    )
    executed = []
    monkeypatch.setattr(
        run_v3_all.run_v3,
        "execute_run",
        lambda **kwargs: executed.append(kwargs),
    )

    outcomes = run_v3_all.execute_all(
        dry_run=True,
        config_path=tmp_path / "config.json",
        artifacts_dir=tmp_path,
    )

    assert outcomes[0]["action"] == "skip_valid"
    assert [item["action"] for item in outcomes[1:]] == ["would_run"] * 8
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
    row = {**_valid_row("L1-DIRECT", 0), **changed}
    monkeypatch.setattr(run_v3_all.run_v3, "load_results", lambda path: [row])

    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        run_v3_all.execute_all(
            config_path=tmp_path / "config.json",
            artifacts_dir=tmp_path,
        )


def test_duplicate_unexpected_and_orphaned_artifacts_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_runner_api,
):
    row = _valid_row("L1-DIRECT", 0)
    monkeypatch.setattr(
        run_v3_all.run_v3,
        "load_results",
        lambda path: [row, row],
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        run_v3_all.execute_all(
            dry_run=True,
            config_path=tmp_path / "config.json",
            artifacts_dir=tmp_path,
        )

    monkeypatch.setattr(
        run_v3_all.run_v3,
        "load_results",
        lambda path: [_valid_row("L9-UNKNOWN", 0)],
    )
    with pytest.raises(RuntimeError, match="unexpected run IDs"):
        run_v3_all.execute_all(
            dry_run=True,
            config_path=tmp_path / "config.json",
            artifacts_dir=tmp_path,
        )

    monkeypatch.setattr(run_v3_all.run_v3, "load_results", lambda path: [])
    smoke_reservation = (
        tmp_path
        / ".run_reservations"
        / "SMOKE-L1-DIRECT-s0-r1.json"
    )
    smoke_reservation.parent.mkdir(parents=True)
    smoke_reservation.write_text('{"run_kind":"smoke"}\n', encoding="utf-8")
    partial = tmp_path / "logs" / "L1-DIRECT-s0-r1.jsonl"
    partial.parent.mkdir(parents=True)
    partial.write_text('{"status":"running"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="unregistered.*artifacts"):
        run_v3_all.execute_all(
            dry_run=True,
            config_path=tmp_path / "config.json",
            artifacts_dir=tmp_path,
        )

    partial.unlink()
    assert run_v3_all._unregistered_artifact_paths(tmp_path, set()) == ()
    unexpected = tmp_path / "checkpoints" / "L9-UNKNOWN-s0-r1.npz"
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_bytes(b"orphan")
    with pytest.raises(RuntimeError, match="unregistered scientific artifacts"):
        run_v3_all.execute_all(
            dry_run=True,
            config_path=tmp_path / "config.json",
            artifacts_dir=tmp_path,
        )


def test_executes_strictly_in_order_without_test_or_parallel_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    public_runner_api,
):
    monkeypatch.setattr(run_v3_all.run_v3, "load_results", lambda path: [])
    calls: list[dict] = []

    def fake_execute(**kwargs):
        calls.append(kwargs)
        return _valid_row(kwargs["config_id"], kwargs["seed"])

    monkeypatch.setattr(run_v3_all.run_v3, "execute_run", fake_execute)
    outcomes = run_v3_all.execute_all(
        config_path=tmp_path / "config.json",
        artifacts_dir=tmp_path,
        verbose=False,
    )

    assert [(call["config_id"], call["seed"]) for call in calls] == list(
        run_v3_all.RUN_PLAN
    )
    assert [item["action"] for item in outcomes] == ["executed"] * 9
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
    monkeypatch.setattr(run_v3_all.run_v3, "load_results", lambda path: [])
    calls = []

    def fail_second(**kwargs):
        calls.append((kwargs["config_id"], kwargs["seed"]))
        if len(calls) == 2:
            raise RuntimeError("training failed")
        return _valid_row(kwargs["config_id"], kwargs["seed"])

    monkeypatch.setattr(run_v3_all.run_v3, "execute_run", fail_second)
    with pytest.raises(RuntimeError, match="training failed"):
        run_v3_all.execute_all(
            config_path=tmp_path / "config.json",
            artifacts_dir=tmp_path,
        )

    assert calls == [("L1-DIRECT", 0), ("L1-DIRECT", 1)]
