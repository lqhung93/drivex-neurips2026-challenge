import numpy as np

from drivex26.baselines import FUTURE_T, constant_acceleration, constant_velocity


def quadratic(t, v, a):
    """Ego path along +x with speed v and acceleration a, at the origin when t = 0."""
    return np.stack([v * t + 0.5 * a * t ** 2, np.zeros_like(t)], axis=-1)


def test_future_times_are_12_steps_at_2hz():
    np.testing.assert_allclose(FUTURE_T, 0.5 * np.arange(1, 13))


def test_cv_extrapolates_constant_speed():
    t = np.array([[-2.0, -1.5, -1.0, -0.5]])
    pred = constant_velocity(quadratic(t, 5.0, 0.0), t)
    np.testing.assert_allclose(pred, quadratic(FUTURE_T, 5.0, 0.0)[None], atol=1e-9)


def test_cv_uses_real_history_timestamps():
    # last history sample 0.4 s before anchor, 2 m behind -> 5 m/s, not 4 m/s
    t = np.array([[-1.9, -1.4, -0.9, -0.4]])
    pred = constant_velocity(quadratic(t, 5.0, 0.0), t)
    np.testing.assert_allclose(pred[0, -1], [30.0, 0.0], atol=1e-9)


def test_ca_is_exact_on_constant_acceleration_with_jittered_timestamps():
    t = np.array([[-2.02, -1.47, -1.01, -0.52]])
    pred = constant_acceleration(quadratic(t, 5.0, 1.0), t)
    np.testing.assert_allclose(pred, quadratic(FUTURE_T, 5.0, 1.0)[None], atol=1e-9)


def test_batched_output_shape():
    t = np.tile([-2.0, -1.5, -1.0, -0.5], (7, 1))
    hist = np.random.default_rng(0).normal(size=(7, 4, 2))
    assert constant_velocity(hist, t).shape == (7, 12, 2)
    assert constant_acceleration(hist, t).shape == (7, 12, 2)
