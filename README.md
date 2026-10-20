# Rollout Diff

Deterministic analysis of behavioral shifts across RL training: rollout ingestion, static
feature extraction, prompt-balanced aggregation, change detection, and supporting evidence.

## Setup

Requires Python 3.11+.

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

## Usage

```python
from rolloutdiff import analyze_rollouts, load_trl

rollouts = load_trl(
    "/path/to/completions", run="experiment-1", reward_column="training_passed"
)
analysis = analyze_rollouts(rollouts)
print(analysis.evidence)
print(analysis.sequential.screening)
```

`evidence` has one row per feature series and `scalar:reward`, including non-alerting and
unscorable series.

| Loader | Accepted input | Schema mapping |
| :--- | :--- | :--- |
| `load_trl` | Completions dir, trainer output dir, or `completions_<step>.parquet` | Uses `reward_column` if given, else `reward` (TRL may log one column per reward function and no total); NaN rewards fail |
| `load_verl` | `<step>.jsonl` or `step_<step>.jsonl`, optionally `.gz` | Maps `input`, `output`, `score`, and `step`; tested on a verl 0.9.1 dump shard. `score` is the summed `token_level_scores` unless the reward function returns its own `score` (DAPO's excludes the overlong penalty) |
| `from_table` | DataFrame | Maps canonical column names to source columns via the `columns` dict |

## Input contracts

* Rows require `run`, nonnegative integer `step`, `sample_id`, `prompt_id`, `prompt`,
  `response`, and finite `reward`; `(run, sample_id)` must be unique. Unsupported message
  content fails.
* `prompt_id` identifies exact prompt content within a run, including chat roles and
  metadata. Generated `sample_id` values depend on file name and row position; pass explicit
  IDs if rows may be reordered.
* Each feature has a tier (`structural` or `prose`) and a kind (`binary` or `continuous`).
  Complete `<think>` and `<thinking>` blocks are stripped first, and Python fragments are
  parsed without execution. Python 3.11 rejects f-strings with a backslash in the expression
  part (PEP 701), so a few rollouts per public run lose AST features there
  (`code_parse_fail`); pinned local values assume 3.12 or later.
* The sparse vocabulary is selected over the full dataset: a token needs $\ge 5\%$
  prompt-balanced prevalence at some `(run, step)`, capped at $2{,}000$ tokens per tier.

## Aggregation and uncertainty

Each step averages completions within each exact `prompt_id` and weights prompts equally;
absent sparse features count as $0$. An optional `sampling_units` DataFrame maps every
`(run, prompt_id)` to a `sampling_unit_id` (default: one unit per prompt). Units cluster
standard errors and keep related prompts in the same replication half without changing point
estimates. For $P$ prompt means $x_p$ with mean $\mu$ and $U$ units, the CR1 standard error is:

$$S_u = \sum_{p \in u} (x_p - \mu), \qquad \text{SE} = \frac{1}{P}\sqrt{\frac{U}{U - 1} \sum_{u=1}^{U} S_u^2}$$

Steps with $U < 2$ have unavailable $\text{SE}$.

## Change detectors

**Sequential (`detect_sequential`)**: The first $B = 15$ observed steps freeze the baseline
mean and $\text{scale} = \max(\text{baseline SD}, \text{median baseline SE}, 10^{-9})$. An
alert needs $3$ consecutive logged steps in one direction with $|z| > 4$ (continuous), or $z > 4$ and
a prevalence rise $\ge 0.10$ (binary). Missing baseline $\text{SE}$ makes a series
unscorable; missing monitoring $\text{SE}$ resets the streak. A baseline with no measurable
variance is flagged `numerical_floor_dominant`, and its alerts depend on input units.

**Retrospective (`detect_retrospective`)**: Selects the best single $L_2$ mean-shift split
with $\ge 5$ steps per segment when
$\text{SSE}_{\text{null}} - \text{SSE}_{\text{split}} > 3 \log(n) \cdot \max(\text{SSE}_{\text{split}} / (n - 2), \text{mean}(\text{SE}^2))$
and the absolute shift is $\ge 0.10$ (binary) or $\ge 0$ (continuous). `changepoint_step` starts the
right segment. With `dependence="ar1"` (default) the split is kept only if a stationary AR(1)
covariance plus measurement variance, profiled separately under the constant and split mean
models, gives $2\Delta\ell > 3 \log(n)$ with the same GLS direction; exact piecewise-constant
zero-SE splits stay selected as `exact_zero_variance_unconfirmed`. Binary series with a
mixed step use quasi-binomial measurement variance $\phi\, m(1-m)/U_t$ at segment mean $m$;
other series use measured $\text{SE}^2$. Missing $\text{SE}$ makes a series unscorable.

## Supporting evidence

Evidence anchors on the sequential alert: reward association and prompt mix use streak onset
through alert, and persistence uses later steps. Retrospective-only changes have no anchored
evidence.

| Metric | Definition | Edge cases |
| :--- | :--- | :--- |
| `split_half_replication` | Both whole-unit halves alert in the full run's direction within $10$ full-grid steps. | `contiguous_prefix` (default) stops at the first gap; `fixed_window` resumes after gaps. An unavailable-$\text{SE}$ row that could hide an earlier alert makes the half unavailable. |
| `event_window_corroboration` | A $3$-step same-direction streak in both halves within $10$ steps of the full alert. | Contradicts only in a complete, scorable window; otherwise unavailable. |
| `persistence` | Fraction of later evaluable steps keeping $\ge 50\%$ of the alert-step shift. | No evaluable later step gives no estimate. |
| `reward_association` | Equal-weighted present-minus-absent reward difference over mixed `(run, step, prompt_id)` groups. | Continuous features return `not_binary`. |
| `prompt_mix` | Positive-prompt breadth and paired change on shared prompts between disjoint windows. | No shared prompts gives no estimate. |

## Limitations

* Error rates are uncalibrated. Results do not establish causality or rank mechanisms;
  lexical indicators, reward association, and prompt mix are descriptive.
* Full-run vocabulary selection is retrospective, and benchmarks score fixed,
  retrospectively selected targets.
* Few sampling units make $\text{SE}$ noisy, and sparse baselines make the sequential scale
  noisy.
* AR(1) confirmation lowers null selections under step dependence and loses power on small
  shifts; `bench.dependence_study` measures both. The binary variance assumes equal unit
  sizes; unequal sizes raise the true variance by $U_t \sum_u (P_u/P)^2$, at most $1.04$ on
  the AISI schedules.
* The retrospective detector fits one sustained shift and is sensitive to outliers and
  transient shifts.

## Benchmarks

Benchmark inputs and outputs stay outside the repository.

* `bench/public_manifest.json` pins $4$ dataset revisions and $118$ Parquet files by size and
  SHA-256; nothing is downloaded. AISI cards declare Apache-2.0, but the underlying problem
  materials may carry other terms. Aria and TRL licensing is unresolved; do not redistribute
  them. Steps with a missing reward or response are dropped whole (AISI2 loses step $403$).
  TRL has one prompt per step, so every series is unscorable and only label attribution is
  evaluated. `--external-root` adds the pinned `rlprobe`, `rldoctor`, and `CHERRL`/`RHDA`
  baselines from `bench/external_manifest.json`.
* `bench.run_local` fits Aria steps $1$ to $15$ (and AISI with `--public-root`), plants
  $+0.20$ prevalence or $+2$ rollout-level SD shifts at step $40$ of $80$, and scores each
  detector on its
  pooled non-degenerate profiles: $\ge 95\%$ of planted shifts found within $5$ steps (alert
  delay or boundary error) and $\le 0.5\%$ of no-change series flagged.

```sh
# --verify-only checks the pins without running analysis
.venv/bin/python -m bench.public_benchmark \
  --data-root /path/to/data --run aisi0 --output /tmp/aisi0.json
.venv/bin/python -m bench.sparse_replication \
  --data-root /path/to/data --run aisi2 --output /tmp/halves.json
.venv/bin/python -m bench.run_local \
  --aria /path/to/aria.parquet --output /tmp/fitted.json
.venv/bin/python -m bench.dependence_study \
  --aria /path/to/aria.parquet --output /tmp/dependence.json
```

## Verification

```sh
.venv/bin/python -m pip install -c .github/ci-constraints.txt ruff
.venv/bin/ruff format --check src bench tests
.venv/bin/ruff check src bench tests
.venv/bin/python -m pytest -q -m 'not local_data'

# Local data for the full suite and the acceptance checks
export ROLLOUTDIFF_ARIA_PATH=/path/to/aria.parquet \
  ROLLOUTDIFF_FULL_ARIA=1 \
  ROLLOUTDIFF_TRL_DIR=/path/to/trl_sunshine \
  ROLLOUTDIFF_PUBLIC_ROOT=/path/to/data \
  ROLLOUTDIFF_VERL_SHARD=/path/to/step_000001.jsonl.gz \
  ROLLOUTDIFF_EXTERNAL_SOURCES=/path/to/snapshots
.venv/bin/python -m pytest -q

# Acceptance checks
.venv/bin/python -m pytest -q -m local_data tests/test_aria_full_pipeline.py tests/test_public_runs.py \
  -k "acceptance or named_target or detection_unscorable"
```

The acceptance tests are ordinary assertions;
`test_local_retrospective_meets_synthetic_acceptance` fails while the retrospective detector
misses its thresholds.
