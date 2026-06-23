import numpy as np
import pytest
from cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor


class TestKNearestNeighbor:
    @pytest.fixture
    def simple_data(self):
        X_train = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        y_train = np.array([0, 1, 1, 0])
        X_test = np.array([
            [0.1, 0.1],
            [0.9, 0.1],
        ])
        return X_train, y_train, X_test

    def test_train_stores_data(self, simple_data):
        X_train, y_train, _ = simple_data
        knn = KNearestNeighbor()
        knn.train(X_train, y_train)
        np.testing.assert_array_equal(knn.X_train, X_train)
        np.testing.assert_array_equal(knn.y_train, y_train)

    def test_distances_two_loops_shape(self, simple_data):
        X_train, y_train, X_test = simple_data
        knn = KNearestNeighbor()
        knn.train(X_train, y_train)
        dists = knn.compute_distances_two_loops(X_test)
        assert dists.shape == (2, 4)

    def test_distances_one_loop_shape(self, simple_data):
        X_train, y_train, X_test = simple_data
        knn = KNearestNeighbor()
        knn.train(X_train, y_train)
        dists = knn.compute_distances_one_loop(X_test)
        assert dists.shape == (2, 4)

    def test_distances_no_loops_shape(self, simple_data):
        X_train, y_train, X_test = simple_data
        knn = KNearestNeighbor()
        knn.train(X_train, y_train)
        dists = knn.compute_distances_no_loops(X_test)
        assert dists.shape == (2, 4)

    def test_all_distance_methods_agree(self):
        np.random.seed(42)
        X_train = np.random.randn(20, 5)
        y_train = np.random.randint(0, 3, 20)
        X_test = np.random.randn(5, 5)
        knn = KNearestNeighbor()
        knn.train(X_train, y_train)
        d2 = knn.compute_distances_two_loops(X_test)
        d1 = knn.compute_distances_one_loop(X_test)
        d0 = knn.compute_distances_no_loops(X_test)
        np.testing.assert_allclose(d2, d1, atol=1e-10)
        np.testing.assert_allclose(d2, d0, atol=1e-10)

    def test_predict_k1(self, simple_data):
        X_train, y_train, X_test = simple_data
        knn = KNearestNeighbor()
        knn.train(X_train, y_train)
        pred = knn.predict(X_test, k=1, num_loops=0)
        assert pred[0] == 0
        assert pred[1] == 1

    def test_predict_k3(self):
        X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        y_train = np.array([0, 0, 1, 1, 1])
        X_test = np.array([[2.5]])
        knn = KNearestNeighbor()
        knn.train(X_train, y_train)
        pred = knn.predict(X_test, k=3, num_loops=0)
        assert pred[0] == 1

    def test_predict_all_loop_modes(self, simple_data):
        X_train, y_train, X_test = simple_data
        knn = KNearestNeighbor()
        knn.train(X_train, y_train)
        p0 = knn.predict(X_test, k=1, num_loops=0)
        p1 = knn.predict(X_test, k=1, num_loops=1)
        p2 = knn.predict(X_test, k=1, num_loops=2)
        np.testing.assert_array_equal(p0, p1)
        np.testing.assert_array_equal(p0, p2)

    def test_invalid_num_loops(self, simple_data):
        X_train, y_train, X_test = simple_data
        knn = KNearestNeighbor()
        knn.train(X_train, y_train)
        with pytest.raises(ValueError):
            knn.predict(X_test, k=1, num_loops=3)

    def test_distances_known_values(self):
        X_train = np.array([[0.0, 0.0]])
        y_train = np.array([0])
        X_test = np.array([[3.0, 4.0]])
        knn = KNearestNeighbor()
        knn.train(X_train, y_train)
        dists = knn.compute_distances_no_loops(X_test)
        np.testing.assert_allclose(dists[0, 0], 5.0, atol=1e-10)

    def test_predict_labels_shape(self, simple_data):
        X_train, y_train, X_test = simple_data
        knn = KNearestNeighbor()
        knn.train(X_train, y_train)
        dists = knn.compute_distances_no_loops(X_test)
        y_pred = knn.predict_labels(dists, k=1)
        assert y_pred.shape == (X_test.shape[0],)
