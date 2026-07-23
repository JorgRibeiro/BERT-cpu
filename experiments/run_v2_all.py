"""Run the twelve scientific Variable 2 executions in a fixed sequence.

This module is deliberately a small orchestrator.  The unit runner
``experiments.run_v2`` remains responsible for protocol validation, training,
artifact persistence and keeping the official Adult test unavailable.

Resume policy
-------------
An existing run is skipped only when its registered row is complete and the
unit runner validates all referenced artifacts.  A duplicate, incomplete,
invalid or orphaned run aborts the batch instead of being overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from experiments import run_v2


CONFIG_ORDER = (
    "S-BETA-1",
    "S-BETA-0.5",
    "S-BETA-2",
    "S-BETA-5",
)
MODEL_SEEDS = (0, 1, 2)
RUN_PLAN = tuple(
    (config_id, seed)
    for config_id in CONFIG_ORDER
    for seed in MODEL_SEEDS
)


def _partial_artifact_paths(artifacts_dir: Path, run_id: str) -> tuple[Path, ...]:
    """Return unit-run artifacts that must not exist without a result row."""
    candidates = (
        artifacts_dir / "logs" / f"{run_id}.jsonl",
        artifacts_dir / "checkpoints" / f"{run_id}.npz",
        artifacts_dir / ".run_reservations" / f"{run_id}.json",
    )
    return tuple(path for path in candidates if path.exists())


def _registered_run(
    rows: Sequence[dict[str, str]],
    run_id: str,
) -> dict[str, str] | None:
    matches = [row for row in rows if row.get("run_id") == run_id]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate result rows for {run_id}")
    return matches[0] if matches else None


def _is_complete_valid(
    row: dict[str, str],
    *,
    config_path: Path,
) -> bool:
    return (
        row.get("status") == "completed_valid"
        and row.get("run_kind") == "scientific"
        and run_v2.result_artifacts_valid(row, config_path=config_path)
    )


def _unit_command(
    *,
    config_id: str,
    seed: int,
    config_path: Path,
    artifacts_dir: Path,
    verbose: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.run_v2",
        "--config-id",
        config_id,
        "--seed",
        str(seed),
        "--repetition",
        "1",
        "--config",
        str(config_path),
        "--artifacts-dir",
        str(artifacts_dir),
    ]
    if not verbose:
        command.append("--quiet")
    return command


def execute_all(
    *,
    dry_run: bool = False,
    config_path: Path = run_v2.DEFAULT_CONFIG,
    artifacts_dir: Path = run_v2.DEFAULT_ARTIFACTS,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Execute or describe the frozen V2 plan, strictly one run at a time."""
    config_path = config_path.resolve()
    artifacts_dir = artifacts_dir.resolve()
    config = run_v2.load_config(config_path)
    run_v2.validate_frozen_config(config)
    results_path = artifacts_dir / "results.csv"
    rows = run_v2.load_results(results_path)
    expected_ids = {
        run_v2.expected_run_id(config_id, seed)
        for config_id, seed in RUN_PLAN
    }
    recorded_ids = [row.get("run_id", "") for row in rows]
    if len(recorded_ids) != len(set(recorded_ids)):
        raise RuntimeError("cannot resume: results.csv contains duplicate result rows")
    unexpected_ids = sorted(set(recorded_ids) - expected_ids)
    if unexpected_ids:
        raise RuntimeError(
            "cannot resume: results.csv contains unexpected run IDs: "
            f"{unexpected_ids}"
        )
    outcomes: list[dict[str, Any]] = []

    for config_id, seed in RUN_PLAN:
        run_id = run_v2.expected_run_id(
            config_id,
            seed,
            repetition=1,
            smoke=False,
        )
        existing = _registered_run(rows, run_id)
        if existing is not None:
            if not _is_complete_valid(existing, config_path=config_path):
                raise RuntimeError(
                    f"cannot resume: registered run is incomplete or invalid: {run_id}"
                )
            outcome = {
                "run_id": run_id,
                "config_id": config_id,
                "seed": seed,
                "action": "skip_valid",
            }
            outcomes.append(outcome)
            print(f"SKIP {run_id}: complete artifacts already validated")
            continue

        partial_paths = _partial_artifact_paths(artifacts_dir, run_id)
        if partial_paths:
            displayed = ", ".join(str(path) for path in partial_paths)
            raise RuntimeError(
                f"cannot resume: unregistered artifacts exist for {run_id}: {displayed}"
            )

        if dry_run:
            outcome = {
                "run_id": run_id,
                "config_id": config_id,
                "seed": seed,
                "action": "would_run",
            }
            outcomes.append(outcome)
            print(f"PLAN {run_id}")
            continue

        result = run_v2.execute_run(
            config_id=config_id,
            seed=seed,
            repetition=1,
            smoke=False,
            config_path=config_path,
            artifacts_dir=artifacts_dir,
            verbose=verbose,
            command=_unit_command(
                config_id=config_id,
                seed=seed,
                config_path=config_path,
                artifacts_dir=artifacts_dir,
                verbose=verbose,
            ),
        )
        if result.get("run_id") != run_id or not _is_complete_valid(
            result,
            config_path=config_path,
        ):
            raise RuntimeError(
                f"unit runner did not return complete valid artifacts for {run_id}"
            )
        rows.append(result)
        outcome = {
            "run_id": run_id,
            "config_id": config_id,
            "seed": seed,
            "action": "executed",
        }
        outcomes.append(outcome)
        print(f"DONE {run_id}")

    return outcomes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all twelve frozen V2 configurations sequentially.",
    )
    parser.add_argument("--config", type=Path, default=run_v2.DEFAULT_CONFIG)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=run_v2.DEFAULT_ARTIFACTS,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate resume state and print the plan without training",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> list[dict[str, Any]]:
    args = _build_parser().parse_args(argv)
    outcomes = execute_all(
        dry_run=args.dry_run,
        config_path=args.config,
        artifacts_dir=args.artifacts_dir,
        verbose=not args.quiet,
    )
    skipped = sum(item["action"] == "skip_valid" for item in outcomes)
    planned = sum(item["action"] == "would_run" for item in outcomes)
    executed = sum(item["action"] == "executed" for item in outcomes)
    print(
        f"V2 batch: {skipped} skipped, {planned} planned, "
        f"{executed} executed"
    )
    return outcomes


if __name__ == "__main__":
    main()
