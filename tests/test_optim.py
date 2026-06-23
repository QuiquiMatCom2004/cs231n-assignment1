import numpy as np
import pytest
from cs231n.optim import sgd, sgd_momentum, rmsprop, adam


class TestSGD:
    def test_basic_update(self):
        w = np.array([1.0, 2.0, 3.0])
        dw = np.array([0.1, 0.2, 0.3])
        config = {"learning_rate": 1.0}
        next_w, _ = sgd(w, dw, config)
        np.testing.assert_allclose(next_w, [0.9, 1.8, 2.7])

    def test_default_config(self):
        w = np.array([1.0])
        dw = np.array([1.0])
        next_w, config = sgd(w, dw)
        assert "learning_rate" in config
        assert config["learning_rate"] == 1e-2

    def test_learning_rate(self):
        w = np.array([10.0])
        dw = np.array([1.0])
        config = {"learning_rate": 0.5}
        next_w, _ = sgd(w, dw, config)
        np.testing.assert_allclose(next_w, [9.5])

    def test_zero_gradient(self):
        w = np.array([5.0, 3.0])
        dw = np.zeros(2)
        next_w, _ = sgd(w, dw)
        np.testing.assert_array_equal(next_w, w)


class TestSGDMomentum:
    def test_basic_update(self):
        w = np.array([1.0, 2.0])
        dw = np.array([0.1, 0.2])
        config = {"learning_rate": 0.01, "momentum": 0.9}
        next_w, config = sgd_momentum(w, dw, config)
        assert next_w.shape == w.shape
        assert "velocity" in config

    def test_velocity_accumulation(self):
        w = np.array([0.0])
        dw = np.array([1.0])
        config = {"learning_rate": 0.1, "momentum": 0.9, "velocity": np.zeros(1)}
        _, config1 = sgd_momentum(w, dw, config)
        v1 = config1["velocity"].copy()
        _, config2 = sgd_momentum(w, dw, config1)
        v2 = config2["velocity"]
        expected_v2 = 0.9 * v1 - 0.1 * 1.0
        np.testing.assert_allclose(v2, expected_v2)

    def test_zero_momentum_equals_sgd(self):
        np.random.seed(0)
        w = np.random.randn(5)
        dw = np.random.randn(5)
        lr = 0.01
        w_sgd, _ = sgd(w.copy(), dw, {"learning_rate": lr})
        w_mom, _ = sgd_momentum(w.copy(), dw, {"learning_rate": lr, "momentum": 0.0})
        np.testing.assert_allclose(w_sgd, w_mom)

    def test_default_config(self):
        w = np.array([1.0])
        dw = np.array([1.0])
        _, config = sgd_momentum(w, dw)
        assert config["momentum"] == 0.9
        assert config["learning_rate"] == 1e-2


class TestRMSProp:
    def test_basic_update(self):
        np.random.seed(0)
        w = np.random.randn(5)
        dw = np.random.randn(5)
        config = {"learning_rate": 0.01, "decay_rate": 0.99, "epsilon": 1e-8}
        next_w, config = rmsprop(w, dw, config)
        assert next_w.shape == w.shape
        assert "cache" in config

    def test_cache_updated(self):
        w = np.array([1.0])
        dw = np.array([2.0])
        config = {"learning_rate": 0.01, "decay_rate": 0.99, "epsilon": 1e-8, "cache": np.zeros(1)}
        _, config = rmsprop(w, dw, config)
        expected_cache = 0.99 * 0.0 + 0.01 * 4.0
        np.testing.assert_allclose(config["cache"], [expected_cache])

    def test_default_config(self):
        w = np.array([1.0])
        dw = np.array([1.0])
        _, config = rmsprop(w, dw)
        assert config["learning_rate"] == 1e-2
        assert config["decay_rate"] == 0.99
        assert config["epsilon"] == 1e-8

    def test_decreases_loss_proxy(self):
        np.random.seed(1)
        w = np.random.randn(10)
        config = None
        for _ in range(100):
            dw = 2 * w
            w, config = rmsprop(w, dw, config)
        assert np.sum(w ** 2) < 1.0


class TestAdam:
    def test_basic_update(self):
        np.random.seed(0)
        w = np.random.randn(5)
        dw = np.random.randn(5)
        next_w, config = adam(w, dw)
        assert next_w.shape == w.shape
        assert config["t"] == 1

    def test_iteration_counter(self):
        w = np.array([1.0])
        dw = np.array([0.1])
        config = None
        for i in range(1, 6):
            w, config = adam(w, dw, config)
            assert config["t"] == i

    def test_m_v_updated(self):
        w = np.array([1.0])
        dw = np.array([2.0])
        config = {
            "learning_rate": 1e-3,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "m": np.zeros(1),
            "v": np.zeros(1),
            "t": 0,
        }
        _, config = adam(w, dw, config)
        expected_m = 0.1 * 2.0
        expected_v = 0.001 * 4.0
        np.testing.assert_allclose(config["m"], [expected_m])
        np.testing.assert_allclose(config["v"], [expected_v])

    def test_default_config(self):
        w = np.array([1.0])
        dw = np.array([1.0])
        _, config = adam(w, dw)
        assert config["beta1"] == 0.9
        assert config["beta2"] == 0.999
        assert config["learning_rate"] == 1e-3

    def test_decreases_loss_proxy(self):
        np.random.seed(2)
        w = np.random.randn(10)
        initial_norm = np.sum(w ** 2)
        config = None
        for _ in range(200):
            dw = 2 * w
            w, config = adam(w, dw, config)
        assert np.sum(w ** 2) < initial_norm
