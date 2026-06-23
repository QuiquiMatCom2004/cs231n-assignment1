import numpy as np
from cs231n.layer_utils import affine_relu_forward, affine_relu_backward
from cs231n.gradient_check import eval_numerical_gradient_array


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class TestAffineReluForward:
    def test_output_shape(self):
        x = np.random.randn(4, 6)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        out, _ = affine_relu_forward(x, w, b)
        assert out.shape == (4, 5)

    def test_output_non_negative(self):
        np.random.seed(0)
        x = np.random.randn(10, 8)
        w = np.random.randn(8, 5)
        b = np.random.randn(5)
        out, _ = affine_relu_forward(x, w, b)
        assert np.all(out >= 0)

    def test_known_values(self):
        x = np.array([[1.0, -1.0]])
        w = np.array([[1.0], [-1.0]])
        b = np.array([0.0])
        out, _ = affine_relu_forward(x, w, b)
        np.testing.assert_allclose(out, [[2.0]])


class TestAffineReluBackward:
    def test_gradient_shapes(self):
        np.random.seed(1)
        x = np.random.randn(4, 6)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(4, 5)
        _, cache = affine_relu_forward(x, w, b)
        dx, dw, db = affine_relu_backward(dout, cache)
        assert dx.shape == x.shape
        assert dw.shape == w.shape
        assert db.shape == b.shape

    def test_numerical_gradient_dx(self):
        np.random.seed(2)
        x = np.random.randn(3, 4)
        w = np.random.randn(4, 5)
        b = np.random.randn(5)
        dout = np.random.randn(3, 5)
        _, cache = affine_relu_forward(x, w, b)
        dx, _, _ = affine_relu_backward(dout, cache)
        dx_num = eval_numerical_gradient_array(
            lambda xx: affine_relu_forward(xx, w, b)[0], x, dout
        )
        assert rel_error(dx, dx_num) < 1e-7

    def test_numerical_gradient_dw(self):
        np.random.seed(3)
        x = np.random.randn(3, 4)
        w = np.random.randn(4, 5)
        b = np.random.randn(5)
        dout = np.random.randn(3, 5)
        _, cache = affine_relu_forward(x, w, b)
        _, dw, _ = affine_relu_backward(dout, cache)
        dw_num = eval_numerical_gradient_array(
            lambda ww: affine_relu_forward(x, ww, b)[0], w, dout
        )
        assert rel_error(dw, dw_num) < 1e-7

    def test_numerical_gradient_db(self):
        np.random.seed(4)
        x = np.random.randn(3, 4)
        w = np.random.randn(4, 5)
        b = np.random.randn(5)
        dout = np.random.randn(3, 5)
        _, cache = affine_relu_forward(x, w, b)
        _, _, db = affine_relu_backward(dout, cache)
        db_num = eval_numerical_gradient_array(
            lambda bb: affine_relu_forward(x, w, bb)[0], b, dout
        )
        assert rel_error(db, db_num) < 1e-7
