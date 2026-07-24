"""Run the nine scientific Variable 3 executions in a fixed sequence.

Existing runs are skipped only after full artifact validation. Duplicate,
incomplete, unexpected or orphaned artifacts abort instead of being replaced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from experiments import run_v3


CONFIG_ORDER = run_v3.CONFIG_ORDER
MODEL_SEEDS = (0, 1, 2)
RUN_PLAN = tuple(
    (config_id, seed)
    for config_id in CONFIG_ORDER
    for seed in MODEL_SEEDS
)


def _partial_artifact_paths(
    artifacts_dir: Path,
    run_id: str,
) -> tuple[Path, ...]:
    candidates = (
        artifacts_dir / "logs" / f"{run_id}.jsonl",
        artifacts_dir / "checkpoints" / f"{run_id}.npz",
        artifacts_dir / ".run_reservations" / f"{run_id}.json",
    )
    return tuple(path for path in candidates if path.exists())


def _unregistered_artifact_paths(
    artifacts_dir: Path,
    registered_ids: set[str],
) -> tuple[Path, ...]:
    """Find every top-level scientific artifact without a registered row."""
    locations = (
        (artifacts_dir / "logs", ".jsonl"),
        (artifacts_dir / "checkpoints", ".npz"),
        (artifacts_dir / ".run_reservations", ".json"),
    )
    unregistered: list[Path] = []
    for directory, suffix in locations:
        if not directory.exists():
            continue
        for path in directory.glob(f"*{suffix}"):
            if (
                path.is_file()
                and not path.stem.startswith("SMOKE-")
                and path.stem not in registered_ids
            ):
                unregistered.append(path)
    return tuple(sorted(unregistered))


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
        and run_v3.result_artifacts_valid(row, config_path=config_path)
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
        "experiments.run_v3",
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
    config_path: Path = run_v3.DEFAULT_CONFIG,
    artifacts_dir: Path = run_v3.DEFAULT_ARTIFACTS,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Execute or describe the frozen V3 plan, one run at a time."""
    config_path = config_path.resolve()
    artifacts_dir = artifacts_dir.resolve()
    config = run_v3.load_config(config_path)
    run_v3.validate_frozen_config(config)
    results_path = artifacts_dir / "results.csv"
    rows = run_v3.load_results(results_path)
    expected_ids = {
        run_v3.expected_run_id(config_id, seed)
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
    unregistered_paths = _unregistered_artifact_paths(
        artifacts_dir,
        set(recorded_ids),
    )
    if unregistered_paths:
        displayed = ", ".join(str(path) for path in unregistered_paths)
        raise RuntimeError(
            "cannot resume: unregistered scientific artifacts exist: "
            f"{displayed}"
        )

    outcomes: list[dict[str, Any]] = []
    for config_id, seed in RUN_PLAN:
        run_id = run_v3.expected_run_id(config_id, seed)
        existing = _registered_run(rows, run_id)
        if existing is not None:
            if not _is_complete_valid(existing, config_path=config_path):
                raise RuntimeError(
                    f"cannot resume: registered run is incomplete or invalid: {run_id}"
                )
            outcomes.append(
                {
                    "run_id": run_id,
                    "config_id": config_id,
                    "seed": seed,
                    "action": "skip_valid",
                }
            )
            print(f"SKIP {run_id}: complete artifacts already validated")
            continue

        partial_paths = _partial_artifact_paths(artifacts_dir, run_id)
        if partial_paths:
            displayed = ", ".join(str(path) for path in partial_paths)
            raise RuntimeError(
                f"cannot resume: unregistered artifacts exist for {run_id}: {displayed}"
            )

        if dry_run:
            outcomes.append(
                {
                    "run_id": run_id,
                    "config_id": config_id,
                    "seed": seed,
                    "action": "would_run",
                }
            )
            print(f"PLAN {run_id}")
            continue

        result = run_v3.execute_run(
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
        outcomes.append(
            {
                "run_id": run_id,
                "config_id": config_id,
                "seed": seed,
                "action": "executed",
            }
        )
        print(f"DONE {run_id}")

    return outcomes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all nine frozen V3 configurations sequentially.",
    )
    parser.add_argument("--config", type=Path, default=run_v3.DEFAULT_CONFIG)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=run_v3.DEFAULT_ARTIFACTS,
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
        f"V3 batch: {skipped} skipped, {planned} planned, "
        f"{executed} executed"
    )
    return outcomes


if __name__ == "__main__":
    main()
