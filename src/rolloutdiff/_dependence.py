"""Profile stationary AR(1) covariance for a fixed boundary.

Each mean model profiles its own covariance, including log determinants;
grid-bound flags report limits.
"""

from __future__ import annotations

import numpy as np

_RHO = np.array([-0.95, -0.5, 0.0, 0.5, 0.8, 0.95, 0.99])
_VARIANCE_MULTIPLIERS = np.array(
    [2.0**-16, 2.0**-12, 2.0**-8, 2.0**-6, 2.0**-4, 2.0**-2, 1.0, 4.0]
)


def _whiten_ar1(
    columns: np.ndarray, errors: np.ndarray, rho: np.ndarray, variance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten data/designs using stationary Kalman innovations, without jitter.

    Covariance is variance * rho**abs(i-j) + diag(errors**2) on row indices;
    ``variance`` is latent marginal variance. Cost is O(rows * models).
    """
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        measured = errors**2
        if np.any((errors > 0) & (measured == 0)):
            raise FloatingPointError("measurement variance underflow")
        state = np.zeros((len(rho), columns.shape[1]))
        predicted_variance = variance.copy()
        innovation_variance = variance * (1 - rho**2)
        if np.any((variance > 0) & (innovation_variance == 0)):
            raise FloatingPointError("innovation variance underflow")
        whitened = np.empty((len(columns), len(rho), columns.shape[1]))
        logdet = np.zeros(len(rho))
        for index, values in enumerate(columns):
            forecast_variance = predicted_variance + measured[index]
            if np.any(forecast_variance <= 0):
                raise FloatingPointError("singular covariance")
            residual = values - state
            whitened[index] = residual / np.sqrt(forecast_variance)[:, None]
            logdet += np.log(forecast_variance)
            gain = predicted_variance / forecast_variance
            state = rho[:, None] * (state + gain[:, None] * residual)
            # Avoid cancellation in P - K*P when SE is tiny.
            predicted_variance = rho**2 * gain * measured[index] + innovation_variance
        if not np.all(np.isfinite(whitened)) or not np.all(np.isfinite(logdet)):
            raise FloatingPointError("non-finite covariance whitening")
    return whitened, logdet


def quasi_binomial_errors(
    prevalence: np.ndarray, errors: np.ndarray, units: np.ndarray, means: np.ndarray
) -> np.ndarray | None:
    """Model-based binary measurement errors sqrt(phi * m (1 - m) / U_t).

    Unit proportions at prevalence p_t vary as phi * p_t (1 - p_t), so the
    measured between-unit variances U_t SE_t**2 of the steps with 0 < p_t < 1
    estimate one dispersion phi, pooled by unit degrees of freedom. Evaluated at
    a mean model's segment means m, an all-agree step inside a segment with
    positive prevalence is not a known zero, while an all-zero segment keeps
    zero variance. Returns None when no step identifies phi.
    """
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        measured = errors**2
        if np.any((errors > 0) & (measured == 0)):
            raise FloatingPointError("measurement variance underflow")
        mixed = (prevalence > 0) & (prevalence < 1)
        if not mixed.any():
            return None
        freedom = units[mixed] - 1
        spread = prevalence[mixed] * (1 - prevalence[mixed])
        dispersion = np.sum(freedom * units[mixed] * measured[mixed]) / np.sum(
            freedom * spread
        )
        return np.sqrt(dispersion * means * (1 - means) / units)


def _profile_ar1(
    y: np.ndarray,
    errors: np.ndarray,
    boundary: int,
    rho: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return -2 log likelihood (up to shared constants) and GLS effects."""
    columns = np.column_stack([np.ones(len(y)), np.arange(len(y)) >= boundary, y])
    whitened, logdet = _whiten_ar1(columns, errors, rho, variance)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        intercept, step, response = (whitened[:, :, index] for index in range(3))
        q0 = intercept / np.linalg.norm(intercept, axis=0)
        residual0 = response - q0 * np.sum(q0 * response, axis=0)
        orthogonal_step = step - q0 * np.sum(q0 * step, axis=0)
        # Reorthogonalization keeps near-collinear mean columns stable.
        orthogonal_step -= q0 * np.sum(q0 * orthogonal_step, axis=0)
        step_norm = np.linalg.norm(orthogonal_step, axis=0)
        q1 = orthogonal_step / step_norm
        coefficient = np.sum(q1 * residual0, axis=0)
        residual1 = residual0 - q1 * coefficient
        null = logdet + np.sum(residual0**2, axis=0)
        alternative = logdet + np.sum(residual1**2, axis=0)
        delta = coefficient / step_norm
        if not all(np.all(np.isfinite(value)) for value in [null, alternative, delta]):
            raise FloatingPointError("non-finite covariance fit")
    return null, alternative, delta


def confirm_ar1(
    y: np.ndarray,
    errors: np.ndarray,
    boundary: int,
    penalty_multiplier: float,
    alternative_errors: np.ndarray | None = None,
) -> dict:
    """Confirm a fixed L2 boundary using caller-normalized data and SE.

    ``errors`` enter the constant-mean model and ``alternative_errors`` (default
    the same) the split-mean model; each profiles the shared latent grid. Gain is
    twice the profiled log likelihood improvement; penalty is dimensionless.
    """
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if alternative_errors is None or np.array_equal(alternative_errors, errors):
            alternative_errors = errors
        reference = max(
            float(np.var(y, ddof=1)),
            float(np.mean(errors**2)),
            float(np.mean(alternative_errors**2)),
        )
        variances = reference * _VARIANCE_MULTIPLIERS
        if reference <= 0 or np.any(variances <= 0):
            raise FloatingPointError("covariance variance reference underflow")
        fits, grid_fits = {}, 0
        for name, model_errors in (
            ("null", errors),
            ("alternative", alternative_errors),
        ):
            if name == "alternative" and alternative_errors is errors:
                fits[name] = fits["null"]
                continue
            rho = np.repeat(_RHO, len(variances))
            variance = np.tile(variances, len(_RHO))
            # A zero-latent-variance model is nonsingular only with positive SEs.
            if np.all(model_errors > 0):
                rho = np.r_[rho, 0.0]
                variance = np.r_[variance, 0.0]
            fits[name] = (
                *_profile_ar1(y, model_errors, boundary, rho, variance),
                rho,
                variance,
            )
            grid_fits += len(rho)
        null, _, _, null_rho, null_variance = fits["null"]
        _, alternative, deltas, alternative_rho, alternative_variance = fits[
            "alternative"
        ]
        null_index, alternative_index = (
            int(np.argmin(null)),
            int(np.argmin(alternative)),
        )
        gain = float(null[null_index] - alternative[alternative_index])
        roundoff = (
            32
            * np.finfo(float).eps
            * max(1.0, abs(null[null_index]), abs(alternative[alternative_index]))
        )
        # Separate measurement variances are not nested, so the split model
        # may fit worse; identical variances must not.
        if gain < -roundoff and alternative_errors is errors:
            raise FloatingPointError("nested covariance fit has negative gain")
        gain = max(0.0, gain)
        penalty = float(penalty_multiplier * np.log(len(y)))
        result = {
            "covariance_status": "confirmed" if gain > penalty else "not_confirmed",
            "covariance_gain": gain,
            "covariance_penalty": penalty,
            "covariance_delta_scaled": float(deltas[alternative_index]),
            "covariance_grid_fits": grid_fits,
        }
        for name, rho, variance, index in [
            ("null", null_rho, null_variance, null_index),
            ("alternative", alternative_rho, alternative_variance, alternative_index),
        ]:
            result.update(
                {
                    f"covariance_{name}_rho": float(rho[index]),
                    f"covariance_{name}_tau_variance_scaled": float(variance[index]),
                    f"covariance_{name}_rho_bound": bool(
                        variance[index] > 0 and rho[index] in (_RHO[0], _RHO[-1])
                    ),
                    f"covariance_{name}_variance_bound": bool(
                        variance[index] in (0.0, variances[0], variances[-1])
                    ),
                }
            )
    return result
