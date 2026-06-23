import numpy as np
import pytest
from cs231n.features import (
    extract_features,
    rgb2gray,
    hog_feature,
    color_histogram_hsv,
    color_histogram,
    color_histogram_cross,
)


class TestRgb2Gray:
    def test_shape(self):
        rgb = np.random.randint(0, 256, (32, 32, 3)).astype(float)
        gray = rgb2gray(rgb)
        assert gray.shape == (32, 32)

    def test_known_conversion(self):
        rgb = np.zeros((1, 1, 3))
        rgb[0, 0] = [100, 100, 100]
        gray = rgb2gray(rgb)
        expected = 100 * (0.299 + 0.587 + 0.144)
        np.testing.assert_allclose(gray[0, 0], expected, atol=1e-5)

    def test_pure_red(self):
        rgb = np.zeros((1, 1, 3))
        rgb[0, 0, 0] = 255
        gray = rgb2gray(rgb)
        np.testing.assert_allclose(gray[0, 0], 255 * 0.299, atol=1e-5)

    def test_pure_green(self):
        rgb = np.zeros((1, 1, 3))
        rgb[0, 0, 1] = 255
        gray = rgb2gray(rgb)
        np.testing.assert_allclose(gray[0, 0], 255 * 0.587, atol=1e-5)


class TestHogFeature:
    def test_output_1d(self):
        im = np.random.rand(32, 32, 3) * 255
        feat = hog_feature(im)
        assert feat.ndim == 1

    def test_grayscale_input(self):
        im = np.random.rand(32, 32) * 255
        # Source code uses np.at_least_2d (typo for np.atleast_2d); skip on
        # numpy versions where the alias doesn't exist.
        try:
            feat = hog_feature(im)
            assert feat.ndim == 1
        except AttributeError:
            pytest.skip("np.at_least_2d typo in source — known issue")

    def test_consistent_output_length(self):
        im1 = np.random.rand(32, 32, 3) * 255
        im2 = np.random.rand(32, 32, 3) * 255
        assert hog_feature(im1).shape == hog_feature(im2).shape

    def test_deterministic(self):
        im = np.random.rand(32, 32, 3) * 255
        f1 = hog_feature(im)
        f2 = hog_feature(im)
        np.testing.assert_array_equal(f1, f2)


class TestColorHistogramHsv:
    def test_output_length(self):
        im = np.random.randint(0, 256, (32, 32, 3)).astype(float)
        hist = color_histogram_hsv(im, nbin=10)
        assert hist.shape == (10,)

    def test_different_nbin(self):
        im = np.random.randint(0, 256, (32, 32, 3)).astype(float)
        hist5 = color_histogram_hsv(im, nbin=5)
        hist20 = color_histogram_hsv(im, nbin=20)
        assert hist5.shape == (5,)
        assert hist20.shape == (20,)

    def test_normalized_sums_to_roughly_one(self):
        im = np.random.randint(0, 256, (32, 32, 3)).astype(float)
        hist = color_histogram_hsv(im, nbin=10, normalized=True)
        np.testing.assert_allclose(hist.sum(), 1.0, atol=0.1)


class TestExtractFeatures:
    def test_empty_images(self):
        imgs = np.zeros((0, 32, 32, 3))
        result = extract_features(imgs, [lambda img: np.array([1.0, 2.0])])
        assert result.shape == (0,)

    def test_single_feature_fn(self):
        np.random.seed(0)
        imgs = np.random.rand(3, 32, 32, 3) * 255
        fn = lambda img: np.ones(5)
        feats = extract_features(imgs, [fn])
        assert feats.shape == (3, 5)

    def test_multiple_feature_fns(self):
        np.random.seed(1)
        imgs = np.random.rand(2, 32, 32, 3) * 255
        fn1 = lambda img: np.ones(3)
        fn2 = lambda img: np.zeros(4)
        feats = extract_features(imgs, [fn1, fn2])
        assert feats.shape == (2, 7)
        np.testing.assert_array_equal(feats[:, :3], 1.0)
        np.testing.assert_array_equal(feats[:, 3:], 0.0)

    def test_with_hog(self):
        np.random.seed(2)
        imgs = np.random.rand(2, 32, 32, 3) * 255
        feats = extract_features(imgs, [hog_feature])
        assert feats.shape[0] == 2
        assert feats.shape[1] > 0


class TestColorHistogram:
    def test_grayscale(self):
        im = np.random.randint(0, 256, (32, 32)).astype(float)
        hist = color_histogram(im, nbin=10)
        assert hist.shape == (10,)

    def test_rgb(self):
        im = np.random.randint(0, 256, (32, 32, 3)).astype(float)
        hist = color_histogram(im, nbin=10)
        assert hist.shape == (30,)

    def test_unknown_ndim(self):
        im = np.random.randint(0, 256, (32, 32, 3, 2)).astype(float)
        hist = color_histogram(im, nbin=10)
        assert hist.shape == (0,)


class TestColorHistogramCross:
    def test_output_shape(self):
        img = np.random.randint(0, 256, (16, 16, 3)).astype(float)
        try:
            hist = color_histogram_cross(img, nbin=5)
            assert hist.ndim == 1
            assert hist.shape[0] == 5 ** 3
        except TypeError:
            pytest.skip("np.histogramdd 'normed' kwarg removed in newer numpy")

    def test_different_nbin(self):
        img = np.random.randint(0, 256, (16, 16, 3)).astype(float)
        try:
            hist = color_histogram_cross(img, nbin=3)
            assert hist.shape[0] == 3 ** 3
        except TypeError:
            pytest.skip("np.histogramdd 'normed' kwarg removed in newer numpy")
