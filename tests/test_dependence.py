"""Covariance confirmation checked against dense linear algebra."""

import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from test_detect import series

from rolloutdiff.retrospective import detect_retrospective


def test_default_reports_covariance_confirmation_without_moving_l2_boundary():
    values = np.r_[np.tile([-0.1, 0.1], 10), np.tile([0.9, 1.1], 10)]
    frame = series(values, se=0.05)
    result = detect_retrospective(frame)
    legacy = detect_retrospective(frame, dependence="independent")
    assert_frame_equal(result.changepoints, legacy.changepoints)
    diag = result.diagnostics.iloc[0]
    assert diag.dependence == "ar1"
    assert diag.independent_selected
    assert diag.independent_reason == "penalized_fit"
    assert diag.covariance_status == "confirmed"
    assert diag.covariance_gain > diag.covariance_penalty
    assert diag.covariance_delta_scaled > 0


@pytest.mark.parametrize(
    "rho,tau,errors",
    [
        (0.8, 0.7, [0.1, 0.3, 0.1, 0.2, 0.4, 0.1]),
        (-0.95, 0.2, [0, 0.1, 0, 0.3, 0.2, 0]),
        (0, 0.7, [0.1, 0.1, 0.3, 0.1, 0.2, 0.4]),
        (0, 0, [0.1, 0.1, 0.3, 0.1, 0.2, 0.4]),
        (0.99, 0.1, [0, 0, 0, 0, 0, 0]),
    ],
)
def test_innovations_match_dense_covariance_including_boundary_transition(
    rho, tau, errors
):
    from rolloutdiff._dependence import _profile_ar1, _whiten_ar1

    y = np.array([0.3, 0.1, -0.2, 1.0, 0.7, 1.2])
    errors = np.array(errors)
    design = np.column_stack([np.ones(len(y)), np.arange(len(y)) >= 3])
    columns = np.column_stack([design, y])
    covariance = float(tau) * float(rho) ** np.abs(
        np.subtract.outer(np.arange(len(y)), np.arange(len(y)))
    )
    covariance += np.diag(errors**2)
    whitened, determinant = _whiten_ar1(
        columns, errors, np.array([rho]), np.array([tau])
    )
    dense_inverse_products = columns.T @ np.linalg.solve(covariance, columns)
    np.testing.assert_allclose(
        whitened[:, 0].T @ whitened[:, 0],
        dense_inverse_products,
        rtol=2e-12,
        atol=2e-12,
    )
    assert determinant[0] == pytest.approx(np.linalg.slogdet(covariance)[1], rel=2e-12)
    null, alternative, delta = _profile_ar1(
        y, errors, 3, np.array([rho]), np.array([tau])
    )
    for model, actual in [(design[:, :1], null[0]), (design, alternative[0])]:
        inverse_design = np.linalg.solve(covariance, model)
        coefficient = np.linalg.solve(
            model.T @ inverse_design, model.T @ np.linalg.solve(covariance, y)
        )
        residual = y - model @ coefficient
        expected = np.linalg.slogdet(covariance)[1] + residual @ np.linalg.solve(
            covariance, residual
        )
        assert actual == pytest.approx(expected, rel=2e-12)
    assert delta[0] == pytest.approx(coefficient[1], rel=2e-12)
    assert alternative[0] <= null[0]


def test_singular_covariance_does_not_acquire_jitter():
    from rolloutdiff._dependence import _whiten_ar1

    with pytest.raises(FloatingPointError):
        _whiten_ar1(np.ones((5, 3)), np.zeros(5), np.array([0.0]), np.array([0.0]))


def test_smooth_drift_is_not_confirmed_as_an_abrupt_change():
    frame = series(np.linspace(0, 1, 80), se=0.01)
    legacy = detect_retrospective(frame, dependence="independent")
    assert len(legacy.changepoints) == 1
    result = detect_retrospective(frame)
    assert result.changepoints.empty
    diag = result.diagnostics.iloc[0]
    assert diag.independent_selected
    assert diag.reason == "covariance_not_confirmed"
    assert diag.covariance_gain <= diag.covariance_penalty
    assert diag.covariance_null_rho_bound and diag.covariance_alternative_rho_bound
    assert diag.candidate_step == legacy.diagnostics.iloc[0].candidate_step


def test_gls_direction_disagreement_cannot_confirm_empirical_direction():
    # The low-SE observations have the opposite direction from the majority.
    # A small explicit penalty isolates the direction guard from the default
    # penalty guard.
    values = np.r_[np.tile([0.0, 0.0, 0.0, 1.0], 25), np.tile([1.0, 1.0, 1.0, 0.0], 25)]
    frame = series(values, se=np.tile([1.0, 1.0, 1.0, 0.01], 50))
    result = detect_retrospective(frame, penalty_multiplier=0.1)
    diag = result.diagnostics.iloc[0]
    assert diag.independent_selected
    assert diag.delta > 0 and diag.covariance_delta_scaled < 0
    assert diag.covariance_gain > diag.covariance_penalty
    assert diag.reason == "covariance_direction_mismatch"
    assert result.changepoints.empty


def test_exact_zero_uncertainty_step_remains_descriptive_without_covariance_claim():
    result = detect_retrospective(series([0.0] * 10 + [1.0] * 10, se=0))
    diag = result.diagnostics.iloc[0]
    assert len(result.changepoints) == 1
    assert diag.covariance_status == "exact_zero_variance_unconfirmed"
    assert np.isnan(diag.covariance_gain)
    assert np.isnan(diag.covariance_penalty)
    assert diag.covariance_grid_fits == 0


@pytest.mark.parametrize("factor", [1e-150, 1e150, -7.0])
def test_covariance_confirmation_preserves_continuous_unit_changes(factor):
    values = np.r_[np.tile([-1.0, 1.0], 10), np.tile([9.0, 11.0], 10)]
    base = detect_retrospective(series(values, se=0.5))
    scaled = detect_retrospective(series(values * factor, se=0.5 * abs(factor)))
    a, b = base.diagnostics.iloc[0], scaled.diagnostics.iloc[0]
    assert a.status == b.status == "change"
    assert b.candidate_step == a.candidate_step
    assert b.covariance_gain == pytest.approx(a.covariance_gain)
    assert b.covariance_delta_scaled == pytest.approx(
        a.covariance_delta_scaled * np.sign(factor)
    )
    assert b.covariance_null_rho == a.covariance_null_rho
    assert b.covariance_alternative_rho == a.covariance_alternative_rho


def test_translation_does_not_enter_covariance_fit():
    values = np.r_[np.tile([-1.0, 1.0], 10), np.tile([9.0, 11.0], 10)]
    base = detect_retrospective(series(values, se=0.5)).diagnostics.iloc[0]
    translated = detect_retrospective(series(values + 1e12, se=0.5)).diagnostics.iloc[0]
    assert translated.covariance_gain == pytest.approx(base.covariance_gain)
    assert translated.covariance_delta_scaled == pytest.approx(
        base.covariance_delta_scaled
    )
    assert translated.delta == base.delta


@pytest.mark.parametrize(
    "errors,expected_fits", [([0.05] * 40, 57), ([0.0] + [0.05] * 39, 56)]
)
def test_zero_latent_variance_is_only_profiled_when_measurement_covariance_is_nonsingular(
    errors, expected_fits
):
    values = np.r_[np.tile([-0.1, 0.1], 10), np.tile([0.9, 1.1], 10)]
    diag = detect_retrospective(series(values, se=errors)).diagnostics.iloc[0]
    assert diag.covariance_grid_fits == expected_fits
    assert diag.covariance_gain >= 0


@pytest.mark.parametrize(
    "values,errors,reason",
    [
        ([0.0] * 20, 0.0, "constant"),
        ([0.0] * 10 + [0.05] * 10, 0.01, "below_effect"),
        ([0.0] * 10 + [1.0] * 10, 10.0, "penalty"),
    ],
)
def test_covariance_confirmation_cannot_add_a_legacy_rejection(values, errors, reason):
    diag = detect_retrospective(
        series(values, se=errors, kind="binary")
    ).diagnostics.iloc[0]
    assert not diag.independent_selected
    assert diag.status == "no_change" and diag.reason == reason
    assert diag.covariance_status == "not_needed"
    assert diag.covariance_grid_fits == 0
    assert np.isnan(diag.covariance_gain)


def test_unavailable_uncertainty_does_not_become_no_change_under_covariance_confirmation():
    frame = series([0.0] * 10 + [1.0] * 10, se=[0.1] * 19 + [np.nan])
    saved = frame.copy(deep=True)
    diag = detect_retrospective(frame).diagnostics.iloc[0]
    assert diag.status == "unavailable_uncertainty"
    assert diag.covariance_status == "unscorable"
    assert diag.independent_selected is None
    assert diag.covariance_grid_fits == 0
    assert_frame_equal(frame, saved)


def test_independent_mode_preserves_original_penalized_fit_fields_and_decisions():
    diag = detect_retrospective(
        series([0.0] * 10 + [1.0] * 10, se=0.1), dependence="independent"
    ).diagnostics.iloc[0]
    assert diag.status == "change" and diag.reason == "penalized_fit"
    assert diag.candidate_step == 11
    assert diag.gain_scaled == pytest.approx(5.0)
    assert diag.penalty_scaled == pytest.approx(3 * np.log(20) * 0.01)
    assert diag.independent_selected
    assert diag.covariance_status == "not_requested"
    assert diag.covariance_grid_fits == 0


def test_underflowed_measurement_variance_is_not_replaced_with_zero_or_jitter():
    with pytest.raises(ValueError, match="numerical overflow or underflow"):
        detect_retrospective(series([0.0] * 10 + [1.0] * 10, se=1e-200))


@pytest.mark.parametrize("dependence", ["arma", "", None, 1])
def test_unknown_covariance_policy_is_rejected(dependence):
    with pytest.raises(ValueError, match="dependence"):
        detect_retrospective(series([0.0] * 20, se=0.1), dependence=dependence)


def test_both_mean_models_profile_the_same_covariance_grid_against_dense_reference():
    from rolloutdiff._dependence import _RHO, _VARIANCE_MULTIPLIERS, confirm_ar1

    y = np.array([0.0, 0.1, -0.2, 0.1, -0.1, 0.1, 1.0, 1.1, 0.8, 1.1, 0.9, 1.1])
    errors = np.array([0.1, 0.2, 0.1, 0.3, 0.1, 0.2] * 2)
    reference = max(np.var(y, ddof=1), np.mean(errors**2))
    covariances = [
        (rho, reference * multiplier)
        for rho in _RHO
        for multiplier in _VARIANCE_MULTIPLIERS
    ] + [(0.0, 0.0)]
    distance = np.abs(np.subtract.outer(np.arange(len(y)), np.arange(len(y))))
    design = np.column_stack([np.ones(len(y)), np.arange(len(y)) >= 6])
    costs = [[], []]
    effects = []
    for rho, variance in covariances:
        covariance = variance * rho**distance + np.diag(errors**2)
        for index, model in enumerate([design[:, :1], design]):
            inverse_design = np.linalg.solve(covariance, model)
            coefficient = np.linalg.solve(
                model.T @ inverse_design, model.T @ np.linalg.solve(covariance, y)
            )
            residual = y - model @ coefficient
            costs[index].append(
                np.linalg.slogdet(covariance)[1]
                + residual @ np.linalg.solve(covariance, residual)
            )
        effects.append(coefficient[1])
    null_index, alternative_index = (int(np.argmin(cost)) for cost in costs)
    assert null_index != alternative_index  # Fixture distinguishes separate profiling.
    result = confirm_ar1(y, errors, 6, 3.0)
    assert result["covariance_gain"] == pytest.approx(
        min(costs[0]) - min(costs[1]), rel=1e-12
    )
    assert result["covariance_delta_scaled"] == pytest.approx(
        effects[alternative_index], rel=1e-12
    )
    for label, index in [("null", null_index), ("alternative", alternative_index)]:
        assert result[f"covariance_{label}_rho"] == covariances[index][0]
        assert result[f"covariance_{label}_tau_variance_scaled"] == pytest.approx(
            covariances[index][1]
        )


def test_heteroskedastic_rejection_does_not_claim_temporal_dependence():
    # The precise observations show no shift; both covariance profiles choose
    # independent measurement noise without a latent temporal component.
    values = [0.0] * 20 + [1.0, 1.0, 1.0, 0.0] * 5
    errors = [0.7, 0.7, 0.7, 0.01] * 10
    result = detect_retrospective(series(values, se=errors))
    diag = result.diagnostics.iloc[0]
    assert diag.independent_selected
    assert result.changepoints.empty
    assert diag.covariance_null_rho == diag.covariance_alternative_rho == 0
    assert diag.covariance_null_tau_variance_scaled == 0
    assert diag.covariance_alternative_tau_variance_scaled == 0
    assert diag.covariance_status == "not_confirmed"
    assert diag.reason == "covariance_not_confirmed"
