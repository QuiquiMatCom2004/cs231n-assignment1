import numpy as np
from cs231n.vis_utils import visualize_grid, vis_grid, vis_nn


class TestVisualizeGrid:
    def test_output_shape_single_channel(self):
        Xs = np.random.randn(4, 8, 8, 1)
        grid = visualize_grid(Xs, padding=1)
        assert grid.ndim == 3
        assert grid.shape[2] == 1

    def test_output_shape_three_channels(self):
        Xs = np.random.randn(4, 8, 8, 3)
        grid = visualize_grid(Xs, padding=1)
        assert grid.ndim == 3
        assert grid.shape[2] == 3

    def test_values_in_range(self):
        Xs = np.random.randn(9, 4, 4, 3)
        grid = visualize_grid(Xs, ubound=255.0, padding=0)
        assert grid.min() >= 0
        np.testing.assert_array_less(grid, 255.0 + 1e-6)

    def test_padding_increases_size(self):
        Xs = np.random.randn(4, 8, 8, 1)
        grid_p0 = visualize_grid(Xs, padding=0)
        grid_p2 = visualize_grid(Xs, padding=2)
        assert grid_p2.shape[0] > grid_p0.shape[0]

    def test_single_image(self):
        Xs = np.random.randn(1, 8, 8, 3)
        grid = visualize_grid(Xs, padding=0)
        assert grid.shape == (8, 8, 3)


class TestVisGrid:
    def test_output_shape(self):
        Xs = np.random.randn(4, 8, 8, 3)
        grid = vis_grid(Xs)
        assert grid.ndim == 3
        assert grid.shape[2] == 3

    def test_normalized_0_to_1(self):
        Xs = np.random.randn(4, 8, 8, 3)
        grid = vis_grid(Xs)
        np.testing.assert_allclose(grid.min(), 0.0, atol=1e-10)
        np.testing.assert_allclose(grid.max(), 1.0, atol=1e-10)

    def test_single_image(self):
        Xs = np.random.randn(1, 4, 4, 3)
        grid = vis_grid(Xs)
        assert grid.ndim == 3


class TestVisNN:
    def test_basic(self):
        imgs = [np.random.randn(4, 4, 3) for _ in range(3)]
        rows = [imgs, imgs]
        grid = vis_nn(rows)
        assert grid.ndim == 3
        assert grid.shape[2] == 3

    def test_normalized_0_to_1(self):
        imgs = [np.random.randn(4, 4, 3) for _ in range(2)]
        rows = [imgs]
        grid = vis_nn(rows)
        np.testing.assert_allclose(grid.min(), 0.0, atol=1e-10)
        np.testing.assert_allclose(grid.max(), 1.0, atol=1e-10)
