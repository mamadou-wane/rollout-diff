"""Analyze dumps, render reports, and inspect selected rollouts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters import load_trl, load_verl
from .html_report import load_report, render_report
from .inspection import resolve_exemplars
from .pipeline import analyze_rollouts
from .report import _check_output_path, _write_text, build_report, write_report


def _terminal(value: object) -> str:
    if value is None:
        return "Unavailable"
    return "".join(
        char if char.isprintable() else ascii(char)[1:-1] for char in str(value)
    )


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        super().error(_terminal(message))


class _Once(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may be supplied only once")
        setattr(namespace, self.dest, values)


def _nonempty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must be nonempty")
    return value


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="rolloutdiff", allow_abbrev=False)
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
    render = commands.add_parser(
        "render", help="render a JSON investigation report", allow_abbrev=False
    )
    render.add_argument("input", type=_nonempty, help="version-1 JSON report")
    render.add_argument("--output", required=True, type=_nonempty, action=_Once)
    inspect = commands.add_parser(
        "inspect", help="inspect selected exemplar responses", allow_abbrev=False
    )
    inspect.add_argument("input", type=_nonempty, help="rollout file or directory")
    inspect.add_argument(
        "--format", required=True, choices=["trl", "verl"], action=_Once
    )
    inspect.add_argument("--run", required=True, type=_nonempty, action=_Once)
    inspect.add_argument("--report", required=True, type=_nonempty, action=_Once)
    inspect.add_argument(
        "--event-rank",
        required=True,
        type=_positive_integer,
        action=_Once,
        help="within-run event rank from the saved report",
    )
    inspect.add_argument(
        "--reward-column", type=_nonempty, action=_Once, help="TRL reward column"
    )
    return parser


def _inspection(event: dict) -> str:
    lines = [
        f"Run: {_terminal(event['run'])}",
        f"Event: {_terminal(event['event_id'])} (rank {event['rank']})",
        f"Representative: {_terminal(event['representative_feature'])}",
        f"Ranking status: {_terminal(event['ranking_status'])}",
        f"Onset: {event['onset_step']} | Alert: {event['alert_step']}",
    ]
    for member in event["members"]:
        lines.extend(
            [
                "",
                f"Feature: {_terminal(member['feature'])}",
                f"Membership status: {_terminal(member['status'])}",
                f"Exemplar records: {_terminal(member['exemplar_status'])}",
            ]
        )
        if not member["exemplars"]:
            lines.append("  No exemplar records in the report.")
        for example in member["exemplars"]:
            lines.append("")
            for field, label in (
                ("status", "Status"),
                ("selection_order", "Selection order"),
                ("run", "Run"),
                ("sample_id", "Sample ID"),
                ("step", "Step"),
                ("prompt_id", "Prompt ID"),
                ("reward", "Reward"),
                ("feature_value", "Feature value"),
                ("selection_reason", "Selection reason"),
                ("reward_deviation", "Reward deviation"),
                ("reward_deviation_status", "Reward deviation status"),
            ):
                lines.append(f"  {label}: {_terminal(example[field])}")
            if example["response"] is None:
                lines.append("  Response: Unavailable (no selected reference).")
            else:
                lines.append("  Response (terminal controls escaped):")
                # Only literal LF creates lines; other controls remain visible escapes.
                lines.extend(
                    f"    {_terminal(line)}" for line in example["response"].split("\n")
                )
    return "\n".join(lines)


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
    if (
        args.command in {"analyze", "inspect"}
        and args.reward_column is not None
        and args.format != "trl"
    ):
        parser.error("--reward-column is supported only with --format trl")
    try:
        if args.command == "inspect":
            report = load_report(args.report)
            rollouts = (
                load_trl(args.input, run=args.run, reward_column=args.reward_column)
                if args.format == "trl"
                else load_verl(args.input, run=args.run)
            )
            event = resolve_exemplars(
                report, rollouts, run=args.run, event_rank=args.event_rank
            )
            print(_inspection(event))
            return 0
        output = Path(args.output)
        _check_output_path(output)
        if args.command == "render":
            report = load_report(args.input)
            _write_text(render_report(report), output)
        else:
            rollouts = (
                load_trl(args.input, run=args.run, reward_column=args.reward_column)
                if args.format == "trl"
                else load_verl(args.input, run=args.run)
            )
            report = build_report(analyze_rollouts(rollouts), run=args.run)
            write_report(report, output)
    except (OSError, ValueError) as exc:
        message = _terminal(exc) if args.command == "inspect" else str(exc)
        print(f"rolloutdiff: error: {message}", file=sys.stderr)
        return 1
    if args.command == "analyze":
        _summary(report)
    else:
        print(
            f"Rendered {sum(len(run['events']) for run in report['runs'])} events "
            f"across {len(report['runs'])} runs."
        )
        for run in report["runs"]:
            print(f"Run {run['run']!r}: {len(run['events'])} events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
