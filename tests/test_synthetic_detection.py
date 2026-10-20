"""Controlled cases using synthetic baseline parameters.

Large planted shifts isolate sustained-onset and direction behavior.
"""

import numpy as np
import pandas as pd

SEED = 20261017
N_CASES = 64
N_STEPS = 80
N_PROMPTS = 64
ROLLOUTS_PER_PROMPT = 4
ONSET = 40
MAX_DELAY = 5


def controlled_series(*, planted):
    """Calculate aggregate means and sample SEs from generated prompt means."""
    rng = np.random.default_rng(SEED if planted else SEED + 1)
    rows = []
    for case in range(N_CASES):
        binary = case < N_CASES // 2
        kind = "binary" if binary else "continuous"
        feature = "flag" if binary else "scalar:reward"
        offsets = rng.normal(0, 0.035 if binary else 0.4, size=N_PROMPTS)
        common = 0.0
        for step in range(N_STEPS):
            common = 0.35 * common + rng.normal(0, 0.003 if binary else 0.02)
            changed = planted and step >= ONSET
            if binary:
                probabilities = np.clip(
                    0.12 + offsets + common + 0.45 * changed, 0.001, 0.999
                )
                values = rng.binomial(
                    1, probabilities[:, None], size=(N_PROMPTS, ROLLOUTS_PER_PROMPT)
                )
            else:
                direction = 1 if case % 2 == 0 else -1
                centers = 0.5 + offsets + common + direction * changed
                values = rng.normal(
                    centers[:, None], 0.25, size=(N_PROMPTS, ROLLOUTS_PER_PROMPT)
                )
            prompt_means = values.mean(axis=1)
            rows.append(
                {
                    "run": f"{kind}-{case:02d}",
                    "step": step,
                    "feature": feature,
                    "tier": "structural",
                    "kind": kind,
                    "mean": prompt_means.mean(),
                    "se": prompt_means.std(ddof=1) / np.sqrt(N_PROMPTS),
                    "n_rollouts": values.size,
                    "n_prompts": N_PROMPTS,
                }
            )
    return pd.DataFrame(rows)


def test_planted_binary_and_reward_changes_alert_within_five_observations():
    from rolloutdiff.detect import detect_sequential

    series = controlled_series(planted=True)
    result = detect_sequential(series)
    alerts = result.alerts
    assert len(alerts) == N_CASES
    assert alerts["run"].is_unique
    assert set(alerts["run"]) == set(series["run"])
    assert alerts["onset_step"].between(ONSET, ONSET + MAX_DELAY).all()
    assert alerts["alert_step"].between(ONSET, ONSET + MAX_DELAY).all()
    assert (alerts["alert_step"] >= alerts["onset_step"]).all()
    assert alerts["run"].str.startswith("binary-").sum() == N_CASES // 2
    assert alerts["run"].str.startswith("continuous-").sum() == N_CASES // 2


def test_unchanged_binary_and_reward_series_have_no_alerts_at_fixed_seed():
    from rolloutdiff.detect import detect_sequential

    result = detect_sequential(controlled_series(planted=False))
    assert result.alerts.empty
    assert len(result.diagnostics) == N_CASES
    assert result.diagnostics["n_scored"].eq(N_STEPS - 15).all()
