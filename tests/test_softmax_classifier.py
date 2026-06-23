import numpy as np
import pytest
from cs231n.classifiers.softmax import softmax_loss_naive, softmax_loss_vectorized
from cs231n.gradient_check import eval_numerical_gradient


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class TestSoftmaxLossNaive:
    def test_loss_positive(self):
        np.random.seed(0)
        W = np.random.randn(10, 5) * 0.001
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 5, 20)
        loss, _ = softmax_loss_naive(W, X, y, 0.0)
        assert loss > 0

    def test_gradient_shape(self):
        np.random.seed(1)
        W = np.random.randn(10, 5) * 0.001
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 5, 20)
        _, dW = softmax_loss_naive(W, X, y, 0.0)
        assert dW.shape == W.shape

    def test_no_reg_loss_near_log_c(self):
        np.random.seed(2)
        D, C, N = 10, 5, 100
        W = np.zeros((D, C))
        X = np.random.randn(N, D)
        y = np.random.randint(0, C, N)
        loss, _ = softmax_loss_naive(W, X, y, 0.0)
        expected = -np.log(1.0 / C)
        np.testing.assert_allclose(loss, expected, atol=0.05)

    def test_regularization_increases_loss(self):
        np.random.seed(3)
        W = np.random.randn(10, 5) * 0.01
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 5, 20)
        loss_no_reg, _ = softmax_loss_naive(W, X, y, 0.0)
        loss_with_reg, _ = softmax_loss_naive(W, X, y, 1.0)
        assert loss_with_reg > loss_no_reg

    def test_numerical_gradient(self):
        np.random.seed(4)
        W = np.random.randn(5, 3) * 0.001
        X = np.random.randn(8, 5)
        y = np.random.randint(0, 3, 8)
        _, dW = softmax_loss_naive(W, X, y, 0.1)
        dW_num = eval_numerical_gradient(
            lambda w: softmax_loss_naive(w, X, y, 0.1)[0], W, verbose=False
        )
        assert rel_error(dW, dW_num) < 1e-5


class TestSoftmaxLossVectorized:
    def test_loss_positive(self):
        np.random.seed(10)
        W = np.random.randn(10, 5) * 0.001
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 5, 20)
        loss, _ = softmax_loss_vectorized(W, X, y, 0.0)
        assert loss > 0

    def test_gradient_shape(self):
        np.random.seed(11)
        W = np.random.randn(10, 5) * 0.001
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 5, 20)
        _, dW = softmax_loss_vectorized(W, X, y, 0.0)
        assert dW.shape == W.shape

    def test_matches_naive(self):
        np.random.seed(12)
        W = np.random.randn(10, 5) * 0.001
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 5, 20)
        reg = 0.05
        loss_n, dW_n = softmax_loss_naive(W, X, y, reg)
        loss_v, dW_v = softmax_loss_vectorized(W, X, y, reg)
        np.testing.assert_allclose(loss_n, loss_v, atol=1e-7)
        np.testing.assert_allclose(dW_n, dW_v, atol=1e-7)

    def test_numerical_gradient(self):
        np.random.seed(13)
        W = np.random.randn(5, 3) * 0.001
        X = np.random.randn(8, 5)
        y = np.random.randint(0, 3, 8)
        _, dW = softmax_loss_vectorized(W, X, y, 0.1)
        dW_num = eval_numerical_gradient(
            lambda w: softmax_loss_vectorized(w, X, y, 0.1)[0], W, verbose=False
        )
        assert rel_error(dW, dW_num) < 1e-5

    def test_regularization_increases_loss(self):
        np.random.seed(14)
        W = np.random.randn(10, 5) * 0.01
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 5, 20)
        loss0, _ = softmax_loss_vectorized(W, X, y, 0.0)
        loss1, _ = softmax_loss_vectorized(W, X, y, 1.0)
        assert loss1 > loss0
