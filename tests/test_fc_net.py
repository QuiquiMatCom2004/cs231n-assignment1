import numpy as np
import pytest
from cs231n.classifiers.fc_net import TwoLayerNet, FullyConnectedNet
from cs231n.gradient_check import eval_numerical_gradient


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# ---------------------------------------------------------------------------
# TwoLayerNet
# ---------------------------------------------------------------------------

class TestTwoLayerNet:
    def test_init_shapes(self):
        net = TwoLayerNet(input_dim=20, hidden_dim=10, num_classes=5)
        assert net.params["W1"].shape == (20, 10)
        assert net.params["b1"].shape == (10,)
        assert net.params["W2"].shape == (10, 5)
        assert net.params["b2"].shape == (5,)

    def test_init_biases_zero(self):
        net = TwoLayerNet(input_dim=20, hidden_dim=10, num_classes=5)
        np.testing.assert_array_equal(net.params["b1"], 0)
        np.testing.assert_array_equal(net.params["b2"], 0)

    def test_forward_scores_shape(self):
        net = TwoLayerNet(input_dim=12, hidden_dim=8, num_classes=4)
        X = np.random.randn(5, 12)
        scores = net.loss(X)
        assert scores.shape == (5, 4)

    def test_loss_positive(self):
        np.random.seed(0)
        net = TwoLayerNet(input_dim=12, hidden_dim=8, num_classes=4, reg=0.0)
        X = np.random.randn(5, 12)
        y = np.random.randint(0, 4, 5)
        loss, grads = net.loss(X, y)
        assert loss > 0
        assert set(grads.keys()) == {"W1", "b1", "W2", "b2"}

    def test_regularization_increases_loss(self):
        np.random.seed(1)
        net0 = TwoLayerNet(input_dim=12, hidden_dim=8, num_classes=4, reg=0.0)
        net1 = TwoLayerNet(input_dim=12, hidden_dim=8, num_classes=4, reg=1.0)
        net1.params = {k: v.copy() for k, v in net0.params.items()}
        net1.reg = 1.0
        X = np.random.randn(5, 12)
        y = np.random.randint(0, 4, 5)
        loss0, _ = net0.loss(X, y)
        loss1, _ = net1.loss(X, y)
        assert loss1 > loss0

    def test_numerical_gradient_W1(self):
        np.random.seed(2)
        net = TwoLayerNet(input_dim=6, hidden_dim=4, num_classes=3, reg=0.05)
        X = np.random.randn(4, 6)
        y = np.random.randint(0, 3, 4)
        _, grads = net.loss(X, y)
        f = lambda w: (setattr_and_return(net, "W1", w), net.loss(X, y)[0])[1]
        dW1_num = eval_numerical_gradient(f, net.params["W1"], verbose=False)
        assert rel_error(grads["W1"], dW1_num) < 1e-5

    def test_numerical_gradient_W2(self):
        np.random.seed(3)
        net = TwoLayerNet(input_dim=6, hidden_dim=4, num_classes=3, reg=0.05)
        X = np.random.randn(4, 6)
        y = np.random.randint(0, 3, 4)
        _, grads = net.loss(X, y)
        f = lambda w: (setattr_and_return(net, "W2", w), net.loss(X, y)[0])[1]
        dW2_num = eval_numerical_gradient(f, net.params["W2"], verbose=False)
        assert rel_error(grads["W2"], dW2_num) < 1e-5


def setattr_and_return(net, key, val):
    net.params[key] = val
    return None


# ---------------------------------------------------------------------------
# FullyConnectedNet
# ---------------------------------------------------------------------------

class TestFullyConnectedNet:
    def test_init_shapes_3_layer(self):
        net = FullyConnectedNet([10, 8], input_dim=20, num_classes=5)
        assert net.params["W1"].shape == (20, 10)
        assert net.params["b1"].shape == (10,)
        assert net.params["W2"].shape == (10, 8)
        assert net.params["b2"].shape == (8,)
        assert net.params["W3"].shape == (8, 5)
        assert net.params["b3"].shape == (5,)

    def test_num_layers(self):
        net = FullyConnectedNet([10, 8, 6], input_dim=20, num_classes=5)
        assert net.num_layers == 4

    def test_forward_scores_shape(self):
        net = FullyConnectedNet([10], input_dim=12, num_classes=4)
        X = np.random.randn(3, 12).astype(np.float64)
        scores = net.loss(X)
        assert scores.shape == (3, 4)

    def test_loss_and_grads(self):
        np.random.seed(10)
        net = FullyConnectedNet([10, 8], input_dim=12, num_classes=4, dtype=np.float64)
        X = np.random.randn(5, 12)
        y = np.random.randint(0, 4, 5)
        loss, grads = net.loss(X, y)
        assert loss > 0
        for i in range(1, net.num_layers + 1):
            assert f"W{i}" in grads
            assert f"b{i}" in grads

    def test_numerical_gradient(self):
        np.random.seed(11)
        net = FullyConnectedNet(
            [6, 5], input_dim=8, num_classes=3, reg=0.05, dtype=np.float64
        )
        X = np.random.randn(4, 8)
        y = np.random.randint(0, 3, 4)
        _, grads = net.loss(X, y)

        for p_name in ["W1", "W2", "W3", "b1", "b2", "b3"]:
            f = lambda val: (
                setattr_and_return(net, p_name, val),
                net.loss(X, y)[0],
            )[1]
            grad_num = eval_numerical_gradient(f, net.params[p_name], verbose=False)
            err = rel_error(grads[p_name], grad_num)
            assert err < 1e-5, f"Gradient check failed for {p_name}: rel_error={err}"

    def test_with_batchnorm(self):
        np.random.seed(12)
        net = FullyConnectedNet(
            [10, 8],
            input_dim=12,
            num_classes=4,
            normalization="batchnorm",
            dtype=np.float64,
        )
        X = np.random.randn(10, 12)
        y = np.random.randint(0, 4, 10)
        loss, grads = net.loss(X, y)
        assert loss > 0
        assert "gamma1" in grads
        assert "beta1" in grads

    def test_with_dropout(self):
        np.random.seed(13)
        net = FullyConnectedNet(
            [10],
            input_dim=12,
            num_classes=4,
            dropout_keep_ratio=0.5,
            seed=42,
            dtype=np.float64,
        )
        X = np.random.randn(5, 12)
        y = np.random.randint(0, 4, 5)
        loss, grads = net.loss(X, y)
        assert loss > 0

    def test_test_mode_no_dropout(self):
        np.random.seed(14)
        net = FullyConnectedNet(
            [10],
            input_dim=12,
            num_classes=4,
            dropout_keep_ratio=0.5,
            seed=42,
            dtype=np.float64,
        )
        X = np.random.randn(5, 12)
        scores = net.loss(X)
        assert scores.shape == (5, 4)

    def test_batchnorm_gradient(self):
        np.random.seed(15)
        net = FullyConnectedNet(
            [6],
            input_dim=8,
            num_classes=3,
            normalization="batchnorm",
            reg=0.0,
            dtype=np.float64,
        )
        X = np.random.randn(10, 8)
        y = np.random.randint(0, 3, 10)
        _, grads = net.loss(X, y)

        for p_name in ["gamma1", "beta1"]:
            f = lambda val: (
                setattr_and_return(net, p_name, val),
                net.loss(X, y)[0],
            )[1]
            grad_num = eval_numerical_gradient(f, net.params[p_name], verbose=False)
            err = rel_error(grads[p_name], grad_num)
            assert err < 1e-4, f"BN gradient check failed for {p_name}: rel_error={err}"
