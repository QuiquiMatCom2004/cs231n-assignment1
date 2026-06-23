import numpy as np
from cs231n.gradient_check import (
    eval_numerical_gradient,
    eval_numerical_gradient_array,
    grad_check_sparse,
)


class TestEvalNumericalGradient:
    def test_linear_function(self):
        f = lambda x: np.sum(3 * x)
        x = np.array([1.0, 2.0, 3.0])
        grad = eval_numerical_gradient(f, x, verbose=False)
        np.testing.assert_allclose(grad, [3.0, 3.0, 3.0], atol=1e-4)

    def test_quadratic_function(self):
        f = lambda x: np.sum(x ** 2)
        x = np.array([1.0, 2.0, 3.0])
        grad = eval_numerical_gradient(f, x, verbose=False)
        np.testing.assert_allclose(grad, 2 * x, atol=1e-4)

    def test_output_shape(self):
        f = lambda x: np.sum(x ** 3)
        x = np.random.randn(4, 5)
        grad = eval_numerical_gradient(f, x, verbose=False)
        assert grad.shape == x.shape

    def test_mixed_function(self):
        f = lambda x: np.sum(np.sin(x))
        x = np.array([0.0, np.pi / 4, np.pi / 2])
        grad = eval_numerical_gradient(f, x, verbose=False)
        expected = np.cos(x)
        np.testing.assert_allclose(grad, expected, atol=1e-4)


class TestEvalNumericalGradientArray:
    def test_identity(self):
        f = lambda x: x.copy()
        x = np.array([1.0, 2.0, 3.0])
        df = np.array([1.0, 1.0, 1.0])
        grad = eval_numerical_gradient_array(f, x, df)
        np.testing.assert_allclose(grad, [1.0, 1.0, 1.0], atol=1e-4)

    def test_scaling(self):
        f = lambda x: 2 * x
        x = np.array([1.0, 2.0])
        df = np.array([1.0, 0.5])
        grad = eval_numerical_gradient_array(f, x, df)
        np.testing.assert_allclose(grad, [2.0, 1.0], atol=1e-4)

    def test_shape_preserved(self):
        f = lambda x: x ** 2
        x = np.random.randn(3, 4)
        df = np.ones_like(x)
        grad = eval_numerical_gradient_array(f, x, df)
        assert grad.shape == x.shape


class TestGradCheckSparse:
    def test_runs_without_error(self, capsys):
        f = lambda x: np.sum(x ** 2)
        x = np.array([1.0, 2.0, 3.0])
        analytic_grad = 2 * x
        grad_check_sparse(f, x, analytic_grad, num_checks=3)
        captured = capsys.readouterr()
        assert "numerical" in captured.out

    def test_correct_gradient_has_low_error(self, capsys):
        f = lambda x: np.sum(x ** 2)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        analytic_grad = 2 * x
        grad_check_sparse(f, x, analytic_grad, num_checks=5)
        captured = capsys.readouterr()
        for line in captured.out.strip().split("\n"):
            parts = line.split("relative error:")
            if len(parts) == 2:
                err = float(parts[1].strip())
                assert err < 1e-4
