"""Analyze and compare dumps, render reports, and inspect selected rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from ._report_validation import validate_report
from .adapters import _TRL_FILE, _VERL_FILE, _files, load_trl, load_verl
from .comparison import compare_rollouts, resolve_comparison_citations
from .comparison_report import (
    comparison_json,
    render_comparison_report,
    validate_comparison_report,
)
from .html_report import render_report
from .inspection import resolve_exemplars
from .investigation import (
    approve_aria_intervention,
    compare_aria_intervention,
    execute_aria_intervention,
    prepare_aria_intervention,
    render_aria_comparison,
)
from .pipeline import analyze_rollouts
from .replay import SandboxConfig, _json_output, replay_aria, serialize_replay
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


def _step_range(value: str) -> tuple[int, int]:
    if not re.fullmatch(r"[0-9]+:[0-9]+", value):
        raise argparse.ArgumentTypeError("use an inclusive START:END step range")
    start, end = map(int, value.split(":"))
    if start > end:
        raise argparse.ArgumentTypeError("range start must not exceed its end")
    return start, end


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="rollout", allow_abbrev=False)
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
    _diff_parser(commands)
    render = commands.add_parser(
        "render", help="render analysis or comparison JSON as HTML", allow_abbrev=False
    )
    render.add_argument(
        "input", type=_nonempty, help="analysis or comparison JSON report"
    )
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
        type=_positive_integer,
        action=_Once,
        help="within-run event rank from the saved report",
    )
    inspect.add_argument(
        "--side",
        choices=["a", "b"],
        action=_Once,
        help="comparison-report side to resolve",
    )
    inspect.add_argument(
        "--feature",
        type=_nonempty,
        action=_Once,
        help="comparison-report feature to inspect",
    )
    inspect.add_argument(
        "--reward-column", type=_nonempty, action=_Once, help="TRL reward column"
    )
    inspect.add_argument(
        "--with-absent",
        action="store_true",
        help="compare each selected exemplar with a feature-absent sibling",
    )
    replay = commands.add_parser(
        "replay-aria",
        help="explicitly replay one pinned Aria completion",
        allow_abbrev=False,
    )
    replay.add_argument("input", type=_nonempty, help="checksum-pinned Aria parquet")
    for name in (
        "run",
        "sample-id",
        "response-sha256",
        "artifacts",
        "docker-host",
        "output",
    ):
        replay.add_argument("--" + name, required=True, type=_nonempty, action=_Once)
    replay.add_argument(
        "--image",
        type=_nonempty,
        action=_Once,
        help="immutable local Python image SHA-256",
    )
    replay.add_argument(
        "--allow-execution",
        action="store_true",
        help="authorize disposable container execution for this completion",
    )
    _investigation_parser(commands)
    return parser


def _diff_parser(commands):
    diff = commands.add_parser(
        "diff",
        help="compare selected windows or two rollout dumps",
        allow_abbrev=False,
        description=(
            "Compare inclusive step ranges A and B in one run, or two TRL/verl dumps. "
            "One dump requires --a and --b in disjoint chronological order. "
            "Two dumps default to each full observed range. Completions are pooled "
            "within each exact prompt across each selected range. Exact prompts "
            "receive equal weight, and both ranges share one feature vocabulary. "
            "Pairing requires exact prompt content, including chat metadata. "
            "Approximate 95% normal intervals use measured standard errors. "
            "Partial prompt overlap leaves the overall interval unavailable. "
            "Available difference estimates and shared-prompt comparisons remain "
            "reported. Differences may reflect prompt composition."
        ),
    )
    diff.add_argument("input", type=_nonempty, help="side A dump (or shared dump)")
    diff.add_argument("other", nargs="?", type=_nonempty, help="optional side B dump")
    diff.add_argument("--format", choices=["trl", "verl"], required=True, action=_Once)
    diff.add_argument("--run", type=_nonempty, required=True, action=_Once)
    diff.add_argument(
        "--other-format",
        choices=["trl", "verl"],
        action=_Once,
        help="side B format; defaults to --format",
    )
    diff.add_argument(
        "--other-run",
        type=_nonempty,
        action=_Once,
        help="side B run name; defaults to --run",
    )
    diff.add_argument("--a", type=_step_range, action=_Once, help="side A START:END")
    diff.add_argument("--b", type=_step_range, action=_Once, help="side B START:END")
    diff.add_argument(
        "--reward-column", type=_nonempty, action=_Once, help="side A TRL reward column"
    )
    diff.add_argument(
        "--other-reward-column",
        type=_nonempty,
        action=_Once,
        help="side B TRL reward column; defaults to reward",
    )
    diff.add_argument(
        "--keyword",
        type=_nonempty,
        action="append",
        default=[],
        help="literal prompt keyword for mix evidence; repeatable",
    )
    diff.add_argument(
        "--output",
        type=_nonempty,
        required=True,
        action=_Once,
        help="new comparison JSON path (never overwritten)",
    )
    diff.add_argument(
        "--html",
        type=_nonempty,
        action=_Once,
        help="optional new self-contained HTML path",
    )


def _investigation_parser(commands):
    investigation = commands.add_parser(
        "investigate-aria",
        help="prepare and compare one Aria intervention",
        allow_abbrev=False,
    )
    actions = investigation.add_subparsers(dest="investigation_action", required=True)
    prepare = actions.add_parser(
        "prepare", help="freeze a control and one exact edit", allow_abbrev=False
    )
    prepare.add_argument("control", type=_nonempty)
    for name in (
        "source",
        "artifacts",
        "edit",
        "hypothesis",
        "predictions",
        "docker-host",
        "output",
    ):
        prepare.add_argument("--" + name, required=True, type=_nonempty, action=_Once)
    approve = actions.add_parser(
        "approve", help="record case-specific execution approval", allow_abbrev=False
    )
    approve.add_argument("case", type=_nonempty)
    for name in ("case-sha256", "variant-sha256"):
        approve.add_argument("--" + name, required=True, type=_nonempty, action=_Once)
    approve.add_argument(
        "--max-containers", required=True, type=_positive_integer, action=_Once
    )
    approve.add_argument("--allow-execution", action="store_true")
    run = actions.add_parser(
        "run", help="consume approval and execute once", allow_abbrev=False
    )
    run.add_argument("case", type=_nonempty)
    for name in ("source", "artifacts"):
        run.add_argument("--" + name, required=True, type=_nonempty, action=_Once)
    run.add_argument("--allow-execution", action="store_true")
    compare = actions.add_parser(
        "compare", help="compare saved evidence without execution", allow_abbrev=False
    )
    compare.add_argument("case", type=_nonempty)
    compare.add_argument("result", type=_nonempty)
    compare.add_argument("--provenance", type=_nonempty, action=_Once)
    for name in ("output", "html"):
        compare.add_argument("--" + name, required=True, type=_nonempty, action=_Once)


def _json_object(path):
    value = _json_output(Path(path).read_bytes())
    if type(value) is not dict:
        raise ValueError("input must be a valid JSON object without duplicate members")
    return value


def _investigation(args):
    if args.investigation_action == "prepare":
        output = Path(args.output)
        _check_output_path(output)
        case = prepare_aria_intervention(
            args.control,
            args.source,
            artifacts=args.artifacts,
            edit=_json_object(args.edit),
            hypothesis=args.hypothesis,
            predictions=_json_object(args.predictions),
            directory=output,
            config=SandboxConfig(docker_host=args.docker_host),
        )
        print(f"Case SHA-256: {_terminal(case['case_sha256'])}")
        print(f"Variant SHA-256: {_terminal(case['variant_response_sha256'])}")
        print(
            "Prepared without execution. Inspect the case and source diff before approval."
        )
        return 0
    if args.investigation_action == "approve":
        approve_aria_intervention(
            args.case,
            case_sha256=args.case_sha256,
            variant_sha256=args.variant_sha256,
            max_containers=args.max_containers,
            execution_authorized=args.allow_execution,
        )
        print("Case-specific approval recorded. Execution has not started.")
        return 0
    if args.investigation_action == "run":
        comparison = execute_aria_intervention(
            args.case,
            args.source,
            artifacts=args.artifacts,
            execution_authorized=args.allow_execution,
        )
        print(
            f"Investigation: {_terminal(comparison['status'])}. Historical claims remain untrusted."
        )
        return 1 if comparison["status"] == "stopped" else 0
    output, html = Path(args.output), Path(args.html)
    if output.resolve() == html.resolve():
        raise ValueError("JSON and HTML output paths must be distinct")
    _check_output_path(output)
    _check_output_path(html)
    comparison = compare_aria_intervention(
        args.case, args.result, provenance=args.provenance
    )
    rendered = render_aria_comparison(comparison)
    serialized = (
        json.dumps(comparison, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    _write_text(serialized, output)
    _write_text(rendered, html)
    print("Comparison saved. Historical claims remain untrusted.")
    return 0


def _comparison(comparison: dict) -> list[str]:
    lines = [
        f"  Matched comparison: {_terminal(comparison['status'])}",
        "  Descriptive same-prompt, same-step comparison.",
    ]
    if comparison["reason"] is not None:
        lines.append(f"  Reason: {_terminal(comparison['reason'])}")
    for field, label in (
        ("n_rollouts", "Group completions"),
        ("n_present", "Feature-present completions"),
        ("n_absent", "Feature-absent completions"),
        ("reward_present", "Present-group mean reward"),
        ("reward_absent", "Absent-group mean reward"),
        ("difference", "Present-minus-absent mean reward"),
    ):
        lines.append(f"  {label}: {_terminal(comparison[field])}")
    counterpart = comparison["counterpart"]
    if counterpart is None:
        lines.append("  Matched response: Unavailable.")
    else:
        for field, label in (
            ("run", "Matched run"),
            ("sample_id", "Matched sample ID"),
            ("step", "Matched step"),
            ("prompt_id", "Matched prompt ID"),
            ("reward", "Matched reward"),
            ("feature_value", "Matched feature value"),
        ):
            lines.append(f"  {label}: {_terminal(counterpart[field])}")
        lines.append("  Matched response (terminal controls escaped):")
        lines.extend(
            f"    {_terminal(line)}" for line in counterpart["response"].split("\n")
        )
    return lines


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
                if field == "reward" and "comparison" in example:
                    label = "Exemplar reward"
                lines.append(f"  {label}: {_terminal(example[field])}")
            if example["response"] is None:
                lines.append("  Response: Unavailable (no selected reference).")
            else:
                lines.append("  Response (terminal controls escaped):")
                # Only literal LF creates lines; other controls remain visible escapes.
                lines.extend(
                    f"    {_terminal(line)}" for line in example["response"].split("\n")
                )
            if "comparison" in example:
                lines.extend(_comparison(example["comparison"]))
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


def _source_files(path, format):
    root = Path(path)
    if format == "trl":
        if root.is_dir() and (root / "completions").is_dir():
            root /= "completions"
        files = _files(root, _TRL_FILE, "completions_*.parquet")
    else:
        files = _files(root, _VERL_FILE, "*.jsonl*")
    records = []
    for file in files:
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append({"sha256": digest, "size_bytes": file.stat().st_size})
    return records


def _comparison_input(path, format, run, reward_column):
    before = _source_files(path, format)
    rows = (
        load_trl(path, run=run, reward_column=reward_column)
        if format == "trl"
        else load_verl(path, run=run)
    )
    if before != _source_files(path, format):
        raise ValueError("comparison source changed while loading")
    return rows, {
        "format": format,
        "files": before,
        "reward_column": (reward_column or "reward") if format == "trl" else None,
    }


def _saved_report(path):
    value = _json_object(path)
    return (
        validate_comparison_report(value)
        if value.get("report_type") == "comparison"
        else validate_report(value)
    )


def _comparison_outputs(outputs):
    # Prepare complete files before publication; remove only our own links on failure.
    with ExitStack() as cleanup:
        staged = []
        for text, destination in outputs:
            directory = cleanup.enter_context(
                TemporaryDirectory(prefix=".rolloutdiff-", dir=destination.parent)
            )
            temporary = Path(directory) / "content"
            _write_text(text, temporary)
            staged.append((temporary, destination))
        published = []
        try:
            for temporary, destination in staged:
                published.append((destination, temporary.stat()))
                os.link(temporary, destination)
        except BaseException:
            for destination, owned in reversed(published):
                try:
                    current = destination.lstat()
                    if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                        destination.unlink()
                except FileNotFoundError:
                    pass
            raise


def _diff(args):
    if args.other is None and any(
        value is not None
        for value in (
            args.other_format,
            args.other_run,
            args.other_reward_column,
        )
    ):
        raise ValueError("other-side options require a second dump")
    if args.other is None and (args.a is None or args.b is None):
        raise ValueError("one-dump comparisons require both --a and --b")
    other_format = args.other_format or args.format
    if args.other_reward_column is not None and other_format != "trl":
        raise ValueError("--other-reward-column requires a TRL second dump")
    output = Path(args.output)
    _check_output_path(output)
    html = Path(args.html) if args.html else None
    if html is not None:
        if output.resolve() == html.resolve():
            raise ValueError("JSON and HTML output paths must differ")
        _check_output_path(html)
    a, a_source = _comparison_input(
        args.input, args.format, args.run, args.reward_column
    )
    b, b_source = None, a_source
    if args.other is not None:
        b, b_source = _comparison_input(
            args.other,
            other_format,
            args.other_run or args.run,
            args.other_reward_column,
        )
    report = compare_rollouts(a, b, a=args.a, b=args.b, keywords=args.keyword)
    for side, source in zip(report["sides"], (a_source, b_source), strict=True):
        side["source"].update(source)
    encoded = comparison_json(report)
    rendered = render_comparison_report(report) if html else None
    outputs = [(encoded, output)]
    if html is not None:
        outputs.append((rendered, html))
    _comparison_outputs(outputs)
    print(
        f"Compared {len(report['features'])} features; "
        f"{report['prompt_mix']['shared_prompts']} shared exact prompts."
    )
    leading = [f for f in report["features"] if f["kind"] == "binary"][:3]
    if leading:
        print(
            "Largest prevalence differences (B - A, proportions): "
            + ", ".join(
                f"{_terminal(f['feature'])} ({_terminal(f['difference']['estimate'])})"
                for f in leading
            )
        )
    return 0


def _inspect_comparison(args, report):
    if args.side is None or args.feature is None:
        raise ValueError("comparison inspection requires --side and --feature")
    if args.event_rank is not None or args.with_absent:
        raise ValueError("--event-rank and --with-absent require an analysis report")
    rows, source = _comparison_input(
        args.input, args.format, args.run, args.reward_column
    )
    expected = next(item for item in report["sides"] if item["side"] == args.side)
    if expected["source"]["files"] and source != {
        key: expected["source"][key] for key in source
    }:
        raise ValueError("comparison source bytes or loader settings do not match")
    resolved = resolve_comparison_citations(
        report, rows, side=args.side, feature=args.feature
    )
    lines = [
        f"Side: {args.side}",
        f"Feature: {_terminal(args.feature)}",
        f"Citations: {_terminal(resolved['status'])}",
    ]
    for citation in resolved["items"]:
        lines.append("")
        for key in (
            "run",
            "sample_id",
            "step",
            "prompt_id",
            "feature_value",
            "reward",
            "selection_reason",
        ):
            lines.append(f"{key}: {_terminal(citation[key])}")
        lines.append("Response (terminal controls escaped):")
        lines.extend(
            f"  {_terminal(line)}" for line in citation["response"].split("\n")
        )
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (
        args.command in {"analyze", "inspect", "diff"}
        and args.reward_column is not None
        and args.format != "trl"
    ):
        parser.error("--reward-column is supported only with --format trl")
    try:
        if args.command == "diff":
            return _diff(args)
        if args.command == "investigate-aria":
            return _investigation(args)
        if args.command == "replay-aria":
            output = Path(args.output)
            _check_output_path(output)
            options = {"docker_host": args.docker_host}
            if args.image is not None:
                options["image"] = args.image
            record = replay_aria(
                args.input,
                run=args.run,
                sample_id=args.sample_id,
                response_sha256=args.response_sha256,
                artifacts=args.artifacts,
                config=SandboxConfig(**options),
                execution_authorized=args.allow_execution,
            )
            _write_text(serialize_replay(record), output)
            print(
                f"Replay: historical={record['historical']['status']}; task={record['task']['status']}; mechanism={record['mechanism_status']}."
            )
            return (
                0
                if record["historical"]["status"] == "reported"
                and record["task"]["determinate"]
                else 1
            )
        if args.command == "inspect":
            report = _saved_report(args.report)
            if report.get("report_type") == "comparison":
                return _inspect_comparison(args, report)
            if args.event_rank is None:
                parser.error("analysis inspection requires --event-rank")
            if args.side is not None or args.feature is not None:
                raise ValueError("--side and --feature require a comparison report")
            rollouts = (
                load_trl(args.input, run=args.run, reward_column=args.reward_column)
                if args.format == "trl"
                else load_verl(args.input, run=args.run)
            )
            event = resolve_exemplars(
                report,
                rollouts,
                run=args.run,
                event_rank=args.event_rank,
                with_absent=args.with_absent,
            )
            print(_inspection(event))
            return 0
        output = Path(args.output)
        _check_output_path(output)
        if args.command == "render":
            report = _saved_report(args.input)
            if report.get("report_type") == "comparison":
                _write_text(render_comparison_report(report), output)
                print(f"Rendered {len(report['features'])} feature comparisons.")
                return 0
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
        message = (
            _terminal(exc)
            if args.command in {"inspect", "diff", "replay-aria", "investigate-aria"}
            else str(exc)
        )
        print(f"rollout: error: {message}", file=sys.stderr)
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
