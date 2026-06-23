import numpy as np
import pytest
from cs231n.layers import (
    affine_forward,
    affine_backward,
    relu_forward,
    relu_backward,
    batchnorm_forward,
    batchnorm_backward,
    dropout_forward,
    dropout_backward,
    softmax_loss,
)
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# ---------------------------------------------------------------------------
# Affine layer
# ---------------------------------------------------------------------------

class TestAffineForward:
    def test_output_shape(self):
        x = np.random.randn(2, 3, 4)
        w = np.random.randn(12, 5)
        b = np.random.randn(5)
        out, _ = affine_forward(x, w, b)
        assert out.shape == (2, 5)

    def test_known_values(self):
        x = np.array([[1.0, 2.0]])
        w = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([0.5, -0.5])
        out, cache = affine_forward(x, w, b)
        expected = np.array([[1.5, 1.5]])
        np.testing.assert_allclose(out, expected)

    def test_cache_contents(self):
        x = np.random.randn(3, 4)
        w = np.random.randn(4, 5)
        b = np.random.randn(5)
        _, cache = affine_forward(x, w, b)
        cx, cw, cb = cache
        np.testing.assert_array_equal(cx, x)
        np.testing.assert_array_equal(cw, w)
        np.testing.assert_array_equal(cb, b)

    def test_higher_dimensional_input(self):
        np.random.seed(0)
        x = np.random.randn(2, 3, 4)
        w = np.random.randn(12, 5)
        b = np.random.randn(5)
        out, _ = affine_forward(x, w, b)
        expected = x.reshape(2, -1) @ w + b
        np.testing.assert_allclose(out, expected, atol=1e-7)


class TestAffineBackward:
    def test_gradient_shapes(self):
        np.random.seed(1)
        x = np.random.randn(10, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(10, 5)
        _, cache = affine_forward(x, w, b)
        dx, dw, db = affine_backward(dout, cache)
        assert dx.shape == x.shape
        assert dw.shape == w.shape
        assert db.shape == b.shape

    def test_numerical_gradient_dx(self):
        np.random.seed(2)
        x = np.random.randn(3, 4)
        w = np.random.randn(4, 5)
        b = np.random.randn(5)
        dout = np.random.randn(3, 5)
        _, cache = affine_forward(x, w, b)
        dx, _, _ = affine_backward(dout, cache)
        dx_num = eval_numerical_gradient_array(lambda xx: affine_forward(xx, w, b)[0], x, dout)
        assert rel_error(dx, dx_num) < 1e-7

    def test_numerical_gradient_dw(self):
        np.random.seed(3)
        x = np.random.randn(3, 4)
        w = np.random.randn(4, 5)
        b = np.random.randn(5)
        dout = np.random.randn(3, 5)
        _, cache = affine_forward(x, w, b)
        _, dw, _ = affine_backward(dout, cache)
        dw_num = eval_numerical_gradient_array(lambda ww: affine_forward(x, ww, b)[0], w, dout)
        assert rel_error(dw, dw_num) < 1e-7

    def test_numerical_gradient_db(self):
        np.random.seed(4)
        x = np.random.randn(3, 4)
        w = np.random.randn(4, 5)
        b = np.random.randn(5)
        dout = np.random.randn(3, 5)
        _, cache = affine_forward(x, w, b)
        _, _, db = affine_backward(dout, cache)
        db_num = eval_numerical_gradient_array(lambda bb: affine_forward(x, w, bb)[0], b, dout)
        assert rel_error(db, db_num) < 1e-7


# ---------------------------------------------------------------------------
# ReLU layer
# ---------------------------------------------------------------------------

class TestReLUForward:
    def test_positive_passthrough(self):
        x = np.array([1.0, 2.0, 3.0])
        out, _ = relu_forward(x)
        np.testing.assert_array_equal(out, x)

    def test_negative_zeroed(self):
        x = np.array([-1.0, -2.0, -3.0])
        out, _ = relu_forward(x)
        np.testing.assert_array_equal(out, np.zeros(3))

    def test_mixed(self):
        x = np.array([-1.0, 0.0, 1.0])
        out, _ = relu_forward(x)
        np.testing.assert_array_equal(out, np.array([0.0, 0.0, 1.0]))

    def test_output_shape(self):
        x = np.random.randn(3, 4, 5)
        out, _ = relu_forward(x)
        assert out.shape == x.shape

    def test_cache_is_input(self):
        x = np.random.randn(4, 5)
        _, cache = relu_forward(x)
        np.testing.assert_array_equal(cache, x)


class TestReLUBackward:
    def test_gradient_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        dout = np.array([1.0, 1.0, 1.0])
        _, cache = relu_forward(x)
        dx = relu_backward(dout, cache)
        np.testing.assert_array_equal(dx, dout)

    def test_gradient_negative(self):
        x = np.array([-1.0, -2.0, -3.0])
        dout = np.array([1.0, 1.0, 1.0])
        _, cache = relu_forward(x)
        dx = relu_backward(dout, cache)
        np.testing.assert_array_equal(dx, np.zeros(3))

    def test_numerical_gradient(self):
        np.random.seed(5)
        x = np.random.randn(10, 10)
        dout = np.random.randn(*x.shape)
        _, cache = relu_forward(x)
        dx = relu_backward(dout, cache)
        dx_num = eval_numerical_gradient_array(lambda xx: relu_forward(xx)[0], x, dout)
        assert rel_error(dx, dx_num) < 1e-7


# ---------------------------------------------------------------------------
# Batch normalization
# ---------------------------------------------------------------------------

class TestBatchnormForward:
    def test_train_output_normalized(self):
        np.random.seed(10)
        N, D = 100, 5
        x = np.random.randn(N, D) * 5 + 3
        gamma = np.ones(D)
        beta = np.zeros(D)
        bn_param = {"mode": "train"}
        out, _ = batchnorm_forward(x, gamma, beta, bn_param)
        np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(out.var(axis=0), 1, atol=0.1)

    def test_train_with_scale_shift(self):
        np.random.seed(11)
        N, D = 200, 3
        x = np.random.randn(N, D)
        gamma = np.array([2.0, 3.0, 4.0])
        beta = np.array([1.0, -1.0, 0.5])
        bn_param = {"mode": "train"}
        out, _ = batchnorm_forward(x, gamma, beta, bn_param)
        np.testing.assert_allclose(out.mean(axis=0), beta, atol=0.1)

    def test_running_stats_updated(self):
        np.random.seed(12)
        N, D = 50, 4
        x = np.random.randn(N, D)
        gamma = np.ones(D)
        beta = np.zeros(D)
        bn_param = {"mode": "train", "running_mean": np.zeros(D), "running_var": np.zeros(D)}
        batchnorm_forward(x, gamma, beta, bn_param)
        assert not np.allclose(bn_param["running_mean"], 0)

    def test_test_mode_uses_running_stats(self):
        np.random.seed(13)
        N, D = 50, 4
        x = np.random.randn(N, D)
        gamma = np.ones(D)
        beta = np.zeros(D)
        running_mean = x.mean(axis=0)
        running_var = x.var(axis=0)
        bn_param = {
            "mode": "test",
            "running_mean": running_mean.copy(),
            "running_var": running_var.copy(),
        }
        out, _ = batchnorm_forward(x, gamma, beta, bn_param)
        expected = (x - running_mean) / np.sqrt(running_var + 1e-5)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_invalid_mode_raises(self):
        x = np.random.randn(5, 3)
        with pytest.raises(ValueError):
            batchnorm_forward(x, np.ones(3), np.zeros(3), {"mode": "invalid"})


class TestBatchnormBackward:
    def test_gradient_dx(self):
        np.random.seed(14)
        N, D = 10, 5
        x = np.random.randn(N, D)
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)
        bn_param = {"mode": "train"}
        _, cache = batchnorm_forward(x, gamma, beta, bn_param)
        dx, dgamma, dbeta = batchnorm_backward(dout, cache)
        assert dx.shape == x.shape
        assert dgamma.shape == gamma.shape
        assert dbeta.shape == beta.shape

    def test_numerical_gradient_dx(self):
        np.random.seed(15)
        N, D = 6, 4
        x = np.random.randn(N, D)
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)
        bn_param = {"mode": "train"}

        _, cache = batchnorm_forward(x, gamma, beta, bn_param)
        dx, _, _ = batchnorm_backward(dout, cache)

        dx_num = eval_numerical_gradient_array(
            lambda xx: batchnorm_forward(xx, gamma, beta, {"mode": "train"})[0], x, dout
        )
        assert rel_error(dx, dx_num) < 1e-5

    def test_numerical_gradient_dgamma(self):
        np.random.seed(16)
        N, D = 6, 4
        x = np.random.randn(N, D)
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)
        bn_param = {"mode": "train"}

        _, cache = batchnorm_forward(x, gamma, beta, bn_param)
        _, dgamma, _ = batchnorm_backward(dout, cache)

        dgamma_num = eval_numerical_gradient_array(
            lambda g: batchnorm_forward(x, g, beta, {"mode": "train"})[0], gamma, dout
        )
        assert rel_error(dgamma, dgamma_num) < 1e-5

    def test_numerical_gradient_dbeta(self):
        np.random.seed(17)
        N, D = 6, 4
        x = np.random.randn(N, D)
        gamma = np.random.randn(D)
        beta = np.random.randn(D)
        dout = np.random.randn(N, D)
        bn_param = {"mode": "train"}

        _, cache = batchnorm_forward(x, gamma, beta, bn_param)
        _, _, dbeta = batchnorm_backward(dout, cache)

        dbeta_num = eval_numerical_gradient_array(
            lambda b: batchnorm_forward(x, gamma, b, {"mode": "train"})[0], beta, dout
        )
        assert rel_error(dbeta, dbeta_num) < 1e-5


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

class TestDropoutForward:
    def test_train_mode_mask_shape(self):
        np.random.seed(20)
        x = np.random.randn(5, 10)
        dp = {"mode": "train", "p": 0.5, "seed": 42}
        out, cache = dropout_forward(x, dp)
        _, mask = cache
        assert mask.shape == x.shape
        assert out.shape == x.shape

    def test_test_mode_identity(self):
        x = np.random.randn(5, 10)
        dp = {"mode": "test", "p": 0.5}
        out, cache = dropout_forward(x, dp)
        np.testing.assert_array_equal(out, x)
        _, mask = cache
        assert mask is None

    def test_train_mode_some_zeroed(self):
        np.random.seed(21)
        x = np.ones((100, 100))
        dp = {"mode": "train", "p": 0.5, "seed": 123}
        out, _ = dropout_forward(x, dp)
        frac_nonzero = np.count_nonzero(out) / out.size
        assert 0.3 < frac_nonzero < 0.7

    def test_inverted_dropout_scaling(self):
        np.random.seed(22)
        x = np.ones((1000, 1000))
        dp = {"mode": "train", "p": 0.8, "seed": 456}
        out, _ = dropout_forward(x, dp)
        np.testing.assert_allclose(out.mean(), 1.0, atol=0.05)

    def test_p_one_keeps_all(self):
        x = np.random.randn(10, 10)
        dp = {"mode": "train", "p": 1.0, "seed": 0}
        out, _ = dropout_forward(x, dp)
        np.testing.assert_allclose(out, x)


class TestDropoutBackward:
    def test_train_mode(self):
        np.random.seed(23)
        x = np.random.randn(5, 10)
        dp = {"mode": "train", "p": 0.7, "seed": 99}
        _, cache = dropout_forward(x, dp)
        dout = np.random.randn(5, 10)
        dx = dropout_backward(dout, cache)
        assert dx.shape == dout.shape

    def test_test_mode_passthrough(self):
        dout = np.random.randn(5, 10)
        cache = ({"mode": "test", "p": 0.5}, None)
        dx = dropout_backward(dout, cache)
        np.testing.assert_array_equal(dx, dout)

    def test_numerical_gradient(self):
        np.random.seed(24)
        x = np.random.randn(4, 6)
        dp = {"mode": "train", "p": 0.8, "seed": 77}
        dout = np.random.randn(4, 6)
        _, cache = dropout_forward(x, dp)
        dx = dropout_backward(dout, cache)
        dx_num = eval_numerical_gradient_array(
            lambda xx: dropout_forward(xx, {"mode": "train", "p": 0.8, "seed": 77})[0], x, dout
        )
        assert rel_error(dx, dx_num) < 1e-7


# ---------------------------------------------------------------------------
# Softmax loss (from layers.py)
# ---------------------------------------------------------------------------

class TestSoftmaxLoss:
    def test_loss_positive(self):
        np.random.seed(30)
        x = np.random.randn(10, 5)
        y = np.random.randint(0, 5, 10)
        loss, _ = softmax_loss(x, y)
        assert loss > 0

    def test_gradient_shape(self):
        np.random.seed(31)
        x = np.random.randn(10, 5)
        y = np.random.randint(0, 5, 10)
        _, dx = softmax_loss(x, y)
        assert dx.shape == x.shape

    def test_perfect_prediction_low_loss(self):
        x = np.zeros((3, 4))
        x[0, 0] = 100.0
        x[1, 1] = 100.0
        x[2, 2] = 100.0
        y = np.array([0, 1, 2])
        loss, _ = softmax_loss(x, y)
        assert loss < 0.01

    def test_numerical_gradient(self):
        np.random.seed(32)
        N, C = 5, 4
        x = np.random.randn(N, C)
        y = np.random.randint(0, C, N)
        _, dx = softmax_loss(x, y)
        dx_num = eval_numerical_gradient(lambda xx: softmax_loss(xx, y)[0], x, verbose=False)
        assert rel_error(dx, dx_num) < 1e-6

    def test_uniform_scores_loss(self):
        N, C = 100, 10
        x = np.zeros((N, C))
        y = np.random.randint(0, C, N)
        loss, _ = softmax_loss(x, y)
        expected = -np.log(1.0 / C)
        np.testing.assert_allclose(loss, expected, atol=1e-5)
