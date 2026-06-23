import numpy as np
import pytest
from cs231n.classifiers.linear_classifier import LinearClassifier, Softmax


class TestLinearClassifier:
    def test_init_no_weights(self):
        clf = LinearClassifier()
        assert clf.W is None

    def test_train_initializes_weights(self):
        np.random.seed(0)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        clf = Softmax()
        clf.train(X, y, num_iters=1, batch_size=10, verbose=False)
        assert clf.W is not None
        assert clf.W.shape == (10, 3)

    def test_train_returns_loss_history(self):
        np.random.seed(1)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        clf = Softmax()
        loss_history = clf.train(X, y, num_iters=20, batch_size=10, verbose=False)
        assert len(loss_history) == 20

    def test_train_loss_decreases(self):
        np.random.seed(2)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)
        clf = Softmax()
        loss_history = clf.train(
            X, y, learning_rate=1e-2, reg=1e-5, num_iters=200,
            batch_size=50, verbose=False,
        )
        first_losses = np.mean(loss_history[:10])
        last_losses = np.mean(loss_history[-10:])
        assert last_losses < first_losses

    def test_predict_shape(self):
        np.random.seed(3)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        clf = Softmax()
        clf.train(X, y, num_iters=10, batch_size=10, verbose=False)
        preds = clf.predict(X)
        assert preds.shape == (50,)

    def test_predict_valid_classes(self):
        np.random.seed(4)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        clf = Softmax()
        clf.train(X, y, num_iters=50, batch_size=20, verbose=False)
        preds = clf.predict(X)
        assert all(0 <= p <= 2 for p in preds)

    def test_verbose_output(self, capsys):
        np.random.seed(5)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 3, 50)
        clf = Softmax()
        clf.train(X, y, num_iters=101, batch_size=10, verbose=True)
        captured = capsys.readouterr()
        assert "iteration" in captured.out


class TestSoftmaxClassifier:
    def test_loss_method(self):
        np.random.seed(6)
        clf = Softmax()
        clf.W = np.random.randn(10, 3) * 0.001
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 3, 20)
        loss, grad = clf.loss(X, y, 0.01)
        assert loss > 0
        assert grad.shape == clf.W.shape

    def test_regularization_effect(self):
        np.random.seed(7)
        clf = Softmax()
        clf.W = np.random.randn(10, 3) * 0.01
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 3, 20)
        loss_no_reg, _ = clf.loss(X, y, 0.0)
        loss_with_reg, _ = clf.loss(X, y, 1.0)
        assert loss_with_reg > loss_no_reg


class TestLinearClassifierBase:
    def test_base_loss_returns_none(self):
        clf = LinearClassifier()
        result = clf.loss(None, None, None)
        assert result is None
