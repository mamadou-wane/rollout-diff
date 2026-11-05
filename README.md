# Rollout Diff

Rollout Diff uses deterministic analysis to detect behavioral shifts across RL training steps. Its ranked reports connect changes in generated code and prose to supporting evidence and selected responses from your rollout dump.

```mermaid
flowchart LR
    A["TRL / verl rollouts"] --> B["Analyze"]
    B --> C["JSON report"]
    C --> D["Render"]
    D --> E["Searchable HTML report"]
```

## Setup

Use Python 3.11 or newer. Pinned benchmark references use Python 3.12+.

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

With `uv`, install into the existing environment:

```sh
uv pip install --python .venv/bin/python -e '.[test]'
```

## Usage

### Local investigation

```sh
.venv/bin/rollout investigate --root /path/to/rollout-data
```

Choose a TRL or verl dump in the browser to analyze it, inspect ranked events and responses, or compare windows. The local app supports macOS and Linux, reads only within the selected root and binds to loopback. Keep its session URL private. Saved replay and intervention evidence can be viewed without executing responses. Installed wheels include the frontend; Node is needed only to rebuild it. Use `--no-browser` to print the URL without opening it.

### Analyze and render

Analyze TRL completions and render the saved report as searchable HTML:

```sh
.venv/bin/rollout analyze /path/to/completions --format trl --run experiment-1 \
  --reward-column training_passed --output /path/to/report.json

.venv/bin/rollout render /path/to/report.json --output /path/to/report.html
```

For verl dumps, use `--format verl` and omit `--reward-column`. Reports contain event evidence and exemplar references, with raw response text kept in the original dump. `render` accepts analysis or comparison JSON. Choose a fresh output path each time.

### Compare windows or dumps

```sh
.venv/bin/rollout diff /path/to/completions --format trl --run experiment-1 \
  --a 80:100 --b 140:160 --output /path/to/diff.json --html /path/to/diff.html
```

Supply a second dump to compare runs. Reports show differences, uncertainty, prompt overlap and citations for each side. See `rollout diff --help` and `rollout inspect --help` for input and inspection options.

### Inspect an event

Inspect the selected responses for the highest-ranked event:

```sh
.venv/bin/rollout inspect /path/to/completions --format trl --run experiment-1 \
  --reward-column training_passed --report /path/to/report.json --event-rank 1
```

Add `--with-absent` to compare binary-feature examples with feature-absent completions from the same run, step, and prompt. Supply complete groups so reward summaries include every completion. Groups with no absent completion receive an unavailable comparison status.

Inspection checks saved identities and metadata. Version-1 analysis reports have no response fingerprint; comparison reports verify source and response fingerprints.

### Python API

```python
from rolloutdiff import analyze_rollouts, load_trl

rollouts = load_trl(
    "/path/to/completions", run="experiment-1", reward_column="training_passed"
)
analysis = analyze_rollouts(rollouts)
print(analysis.event_ranking)
print(analysis.evidence)
```

`analysis.evidence` covers every extracted feature series plus `scalar:reward`. Use `build_report(analysis)` and `render_report(report)` to create reports in Python, or `resolve_exemplars()` to retrieve selected responses.

Use `compare_rollouts(a, b)` for two dumps or `compare_rollouts(rollouts, a=(80, 100), b=(140, 160))` for two ranges. `comparison_json()` and `render_comparison_report()` produce reports. `resolve_comparison_citations()` retrieves the cited responses. Optional sampling-unit maps account for related prompts.

### Aria replay

Replay a selected Aria completion in disposable Docker containers. This executes generated code and requires the supported Python image installed locally, along with the pinned evaluator and task artifacts.

Set `RESPONSE_SHA256` to the selected response's SHA-256 hash.

```sh
.venv/bin/rollout replay-aria /path/to/aria.parquet --run aria \
  --sample-id step111_row233 --response-sha256 "$RESPONSE_SHA256" \
  --artifacts /path/to/pinned-artifacts --docker-host unix:///var/run/docker.sock \
  --allow-execution --output /path/to/replay.json
```

The JSON result includes host-side task-test comparisons and historical evaluator claims marked as untrusted.

The opt-in `investigate-aria` workflow supports preparing and comparing controlled interventions against frozen Aria replay evidence. Execution requires explicit, single-use authorization. Run `rollout investigate-aria --help` for available commands.

## Inputs

| Loader | Accepted input |
| :--- | :--- |
| `load_trl` | Completions directory, trainer output directory, or `completions_<step>.parquet` |
| `load_verl` | Rollout directory, `<step>.jsonl`, or `step_<step>.jsonl`, optionally gzipped |
| `from_table` | DataFrame, with source-column mapping through `columns` |

TRL uses `reward` by default or a specified `reward_column`. The verl adapter maps `input`, `output`, `score`, and `step`. Custom reward functions can change what `score` represents.

Canonical rows contain `run`, `step`, `sample_id`, `prompt_id`, `prompt`, `response`, and `reward`. Each `(run, sample_id)` pair must be unique, with a nonnegative integer step and a finite reward. Use explicit sample IDs when reordering source rows. A `prompt_id` identifies exact prompt content within a run, including chat roles and metadata.

## Reading the results

`analyze` averages completions within each exact prompt at each step, then weights prompts equally. `diff` pools each prompt's completions across each selected range, then weights exact prompts equally. At least two sampling units are needed to estimate standard errors. Related prompts can share a unit through an optional sampling-unit mapping.

Sequential detection uses the first 15 observed steps as its baseline and requires three consecutive qualifying observations in the same direction. Retrospective detection looks for a single mean shift across the run. See [`detect.py`](src/rolloutdiff/detect.py) and [`retrospective.py`](src/rolloutdiff/retrospective.py) for thresholds and statistical details.

Event ranks provide a heuristic inspection order. Evidence for sequential alerts covers replication, persistence, reward association, and prompt mix. Check availability statuses and numerical-floor warnings when interpreting a result.

Detector error rates are uncalibrated. Vocabulary selection uses the full run for analysis and the selected ranges for comparison. Public benchmark targets were chosen after inspecting the runs. Causal and mechanism claims require independent evidence.

## Development

```sh
.venv/bin/python -m pip install -c .github/ci-constraints.txt ruff
.venv/bin/ruff format --check src bench tests
.venv/bin/ruff check src bench tests
.venv/bin/python -m pytest -q -m 'not local_data'
```

Frontend checks use Node 22.23.1:

```sh
npm ci --prefix frontend
npm run --prefix frontend lint
npm run --prefix frontend typecheck
npm test --prefix frontend
npm run --prefix frontend build
```

The build updates the packaged assets in `src/rolloutdiff/_web`. Commit them with frontend changes. `npm run --prefix frontend test:e2e` exercises the installed application with Chromium and synthetic data. Install that test browser with `cd frontend && npx playwright install chromium`.

See [`bench/`](bench/) for public-run and synthetic benchmarks, and [`tests/`](tests/) for local-data checks. Keep datasets and generated outputs outside the repository. Aria and TRL redistribution permissions remain unresolved.
