# Rollout Diff

Deterministic analysis of behavioral shifts across RL training: rollout ingestion, static
feature extraction, prompt-balanced aggregation, change detection, and supporting evidence.

## Quick Start

Requires Python 3.11+.

```sh
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

```python
from rolloutdiff import analyze_rollouts, load_trl

rollouts = load_trl(
    "/path/to/completions", run="experiment-1", reward_column="training_passed"
)
analysis = analyze_rollouts(rollouts)
print(analysis.evidence)
print(analysis.sequential.screening)
```

| Loader | Accepted Input | Schema Mapping |
| :--- | :--- | :--- |
| `load_trl` | Completions dir, trainer output dir, or `completions_<step>.parquet` | Auto-selects `reward`; other reward columns require `reward_column` |
| `load_verl` | Numeric `<step>.jsonl` files | Expects `input`, `output`, `score`, and `step` |
| `from_table` | DataFrame | Maps canonical column names to source columns via `columns` dict |

## Data and Feature Contracts

* **Canonical Schema**: Rows require `run`, nonnegative integer `step`, `sample_id`,
  `prompt_id`, `prompt`, `response`, and finite `reward`. `(run, sample_id)` must be unique.
  Missing rewards fail; external labels are excluded.
* **Identities**: `prompt_id` identifies exact prompt content within a run, including chat
  roles and metadata. Adapters may generate content hashes; caller-provided IDs are supported.
  Auto-generated `sample_id` values depend on filename and row position; pass explicit IDs if
  reordering rows. Text, supported chat messages, and tool calls are preserved; unsupported
  content fails.
* **Preprocessing**: Complete `<think>` and `<thinking>` blocks are stripped before feature
  extraction, while `resp_chars` measures raw unstripped length. Features declare a tier
  (`structural` or `prose`) and kind (`binary` or `continuous`).
* **Static Code Parsing**: Python fragments are parsed without execution. File-declaration
  extraction supports balanced nested tags up to 64 scopes and ignores declarations in
  literals or comments, across fence boundaries, or without closing tags. Complete children of
  unclosed parents remain usable. Features indicate syntax, not execution.
* **Vocabulary**: Selected across the full supplied dataset. Sparse tokens require $\ge 5\%$
  prompt-balanced prevalence at some `(run, step)` and cap at $2{,}000$ tokens per tier
  (lexical tie-breaking). Both limits are configurable.

## Aggregation and Uncertainty

At each step, completions are averaged within each exact `prompt_id`, and prompts are weighted
equally. Absent sparse features contribute $0$; uncomputable uncertainty remains missing.
Joins use rollout identities, and schedules retain prompt IDs and completion counts.

An optional `sampling_units` DataFrame maps `(run, prompt_id)` to a nonblank
`sampling_unit_id` (1-to-1 coverage required; defaults to one unit per prompt). Grouping
related prompt variants into a shared unit clusters standard errors without altering canonical
prompt IDs or point-estimate weights.

For $P$ prompt means $x_p$, $U$ sampling units, and step mean $\mu = \text{mean}(x_p)$, the
cluster-robust (CR1) standard error is:

$$S_u = \sum_{p \in u} (x_p - \mu), \qquad \text{SE} = \frac{1}{P}\sqrt{\frac{U}{U - 1} \sum_{u=1}^{U} S_u^2}$$

When $U = P$, $\text{SE}$ reduces to $\text{SD} / \sqrt{P}$. Steps with $U < 2$ yield
unavailable $\text{SE}$ (small $U$ remains noisy); zero measured variance yields $0$.

`analyze_rollouts` builds one vocabulary, aggregates the full run and two whole-unit halves
(`crc32(unit_id.encode("utf-8")) & 1`) once, and screens all retained features plus `reward`.
It outputs `evidence` (including non-alerting and unscorable series) and detail tables:
`replication_halves`, `corroboration_halves`, `persistence_steps`, `reward_groups`,
`prompt_values`, `prompt_shifts`, and `half_assignments`.

## Change Detectors

### Sequential (`detect_sequential`)

Freezes the first $B = 15$ observed step means in logged order and scores subsequent steps
using:

$$\text{scale} = \max(\text{baseline SD}, \text{median baseline SE}, 10^{-9}), \qquad z = \frac{\text{step mean} - \text{baseline mean}}{\text{scale}}$$

* **Alert Rules**: Requires $3$ consecutive steps in the same direction with $|z| > 4$
  (continuous) or $z > 4$ and mean increase $\ge 0.10$ within $4\epsilon_{\text{float64}}$
  (binary).
* **State Resets**: Missing baseline $\text{SE}$ marks a series unscorable. Sub-threshold
  steps, sign flips, or missing monitoring $\text{SE}$ reset the streak. Current-step
  $\text{SE}$ never alters the frozen `scale`. Only the first alert is emitted; later scores
  remain available. Near-zero variance dominated by the $10^{-9}$ floor depends on input
  units.

### Retrospective (`detect_retrospective`)

Finds the best single $L_2$ mean-shift split (`changepoint_step` starts the right segment)
with $\ge 5$ steps per segment. Squared costs are normalized; missing $\text{SE}$ is
unscorable.

1. **$L_2$ Screen**: Requires
   $\text{SSE}_{\text{null}} - \text{SSE}_{\text{split}} > 3 \log(n) \cdot \max(\text{SSE}_{\text{split}} / (n - 2), \text{mean}(\text{SE}^2))$
   and an absolute shift $\ge 0.10$ (binary) or $\ge 0$ (continuous, default) on the winning
   split.
2. **AR(1) Confirmation (`dependence="ar1"`, default)**: Profiles both constant-mean and
   split-mean models over a stationary latent AR(1) covariance plus measured step
   $\text{SE}^2$ (no diagonal jitter). Confirms the fixed $L_2$ boundary if
   $2\Delta\ell > 3 \log(n)$ with matching GLS sign. Pass `dependence="independent"` to skip
   AR(1) profiling. Exact piecewise-constant zero-$\text{SE}$ series return
   `exact_zero_variance_unconfirmed`.
   * **Grid**: $\rho \in (-0.95, -0.5, 0, 0.5, 0.8, 0.95, 0.99)$; latent variance multiplies
     $\max(\text{sample variance}, \text{mean}(\text{SE}^2))$ by
     $(2^{-16}, 2^{-12}, 2^{-8}, 2^{-6}, 2^{-4}, 2^{-2}, 1, 4)$, plus $0$ when all
     $\text{SE} > 0$.

## Supporting Evidence and Limitations

Integrated evidence anchors on sequential alerts (reward and prompt mix evaluate streak onset
through alert; persistence evaluates post-alert; retrospective-only changes leave integrated
evidence unavailable).

| Metric | Definition | Edge Cases |
| :--- | :--- | :--- |
| `split_half_replication` | Both halves alert in the same direction as the full run within a span of $10$ full-grid steps. | `contiguous_prefix` (default) stops at the first gap. `fixed_window` requires $\ge \max(2, \lceil B/2 \rceil)$ finite-$\text{SE}$ baseline steps and resumes after gaps (earlier hidden compatible alerts mark the first alert unavailable). Opposing alerts fail. |
| `event_window_corroboration` | Sustained $3$-step same-direction streak in both halves within radius $10$ of the full alert (post-baseline, using `fixed_window`). | Streaks cannot span gaps. Contradicts only if a non-matching window is complete and scorable; otherwise unavailable. Discloses opposite/mixed witnesses. |
| `persistence` | Post-anchor fraction of evaluable steps retaining $\ge 50\%$ of the initial signed shift. | Missing future steps yield no estimate; step coverage is reported separately. |
| `reward_association` | Equal-weighted present-minus-absent reward delta across mixed `(run, step, prompt_id)` groups. | Coverage includes pure groups. Continuous features return `not_binary`. |
| `prompt_mix` | Compares disjoint windows: binary breadth (distinct post-window positive prompts) and paired shared change (CR1 $\text{SE}$ by sampling unit). | No shared prompts yields no estimate. Includes length KS distance, keyword prevalence, and optional difficulty-tag total variation. |

**Limitations**: Error rates are uncalibrated, and results do not establish causality or rank
mechanisms. Full-data vocabulary selection, sparse baselines, small unit counts $U$, temporal
dependence, and transient shifts constrain interpretation.

## Benchmarks and Verification

Generated data and outputs remain outside the repository. Benchmarks test fixed,
retrospectively selected targets rather than open-ended discovery.

* **Public & Restricted Data (`bench/public_manifest.json`)**: Pins $4$ dataset revisions and
  $118$ Parquet files by size and SHA-256 before decoding (no network downloads). AISI dataset
  cards declare Apache-2.0; underlying problem materials may have separate terms. `Aria` and
  `TRL` have unresolved licensing and must not be redistributed. Missing rewards or responses
  drop the entire batch (`AISI2` drops step $403$). `TRL` uses one prompt per step
  ($\text{SE}$ unavailable).
* **Synthetic Plant**: Fits baseline windows using beta-binomial or Gaussian prompt variation
  plus stationary AR(1) step noise, reporting sequential delay and retrospective boundary
  error under fixed seeds and effect sizes.
* **External Baselines (`bench/external_manifest.json`)**: Wraps pinned `rlprobe`, `rldoctor`,
  and `CHERRL`/`RHDA` files under `--external-root` (rejects unlisted Python/bytecode files,
  approximates tokens as $\text{chars}/4$).

```sh
# Public and local benchmarks (--verify-only checks hashes without running analysis)
PYTHONPATH=src .venv/bin/python -m bench.public_benchmark \
  --data-root /path/to/data --run aisi0 --output /tmp/aisi0.json
PYTHONPATH=src .venv/bin/python -m bench.sparse_replication \
  --data-root /path/to/data --run aisi2 --output /tmp/halves.json
PYTHONPATH=src .venv/bin/python -m bench.run_local \
  --aria /path/to/aria.parquet --output /tmp/fitted.json
PYTHONPATH=src .venv/bin/python -m bench.dependence_study \
  --aria /path/to/aria.parquet --output /tmp/dependence.json

# Integration tests and linting
ROLLOUTDIFF_ARIA_PATH=/path/to/aria.parquet \
ROLLOUTDIFF_FULL_ARIA=1 \
ROLLOUTDIFF_TRL_DIR=/path/to/trl_sunshine \
ROLLOUTDIFF_EXTERNAL_SOURCES=/path/to/snapshots \
  .venv/bin/python -m pytest

ruff format --check src bench tests && ruff check src bench tests
```
