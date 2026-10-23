"""Analyze an existing rollout dump and write an investigation report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters import load_trl, load_verl
from .pipeline import analyze_rollouts
from .report import _check_output_path, build_report, write_report


class _Once(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may be supplied only once")
        setattr(namespace, self.dest, values)


def _nonempty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must be nonempty")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rolloutdiff", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser(
        "analyze", help="analyze a TRL or verl rollout dump", allow_abbrev=False
    )
    analyze.add_argument("input", type=_nonempty, help="rollout file or directory")
    analyze.add_argument(
        "--format", required=True, choices=["trl", "verl"], action=_Once
    )
    analyze.add_argument("--run", required=True, type=_nonempty, action=_Once)
    analyze.add_argument("--output", required=True, type=_nonempty, action=_Once)
    analyze.add_argument(
        "--reward-column", type=_nonempty, action=_Once, help="TRL reward column"
    )
    return parser


def _summary(report: dict) -> None:
    for run in report["runs"]:
        counts = run["counts"]
        print(
            f"Run {run['run']}: {counts['screened_series']} screened, "
            f"{counts['sequential_alerts']} alerted; "
            f"{counts['behavioral_events']} behavioral and "
            f"{counts['contextual_events']} contextual events."
        )
        leading = [
            event for event in run["events"] if event["ranking_status"] == "behavioral"
        ][:3]
        if leading:
            print(
                "Leading representatives: "
                + ", ".join(
                    f"{event['rank']}. {event['representative_feature']}"
                    for event in leading
                )
            )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.reward_column is not None and args.format != "trl":
        parser.error("--reward-column is supported only with --format trl")
    try:
        output = Path(args.output)
        _check_output_path(output)
        rollouts = (
            load_trl(args.input, run=args.run, reward_column=args.reward_column)
            if args.format == "trl"
            else load_verl(args.input, run=args.run)
        )
        report = build_report(analyze_rollouts(rollouts), run=args.run)
        write_report(report, output)
    except (OSError, ValueError) as exc:
        print(f"rolloutdiff: error: {exc}", file=sys.stderr)
        return 1
    _summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
