import os
import pickle
import tempfile
import numpy as np
import pytest
from cs231n.data_utils import load_pickle, imread


class TestLoadPickle:
    def test_loads_dict(self, tmp_path):
        data = {"key": "value", "num": 42}
        fpath = tmp_path / "test.pkl"
        with open(fpath, "wb") as f:
            pickle.dump(data, f)
        with open(fpath, "rb") as f:
            loaded = load_pickle(f)
        assert loaded == data

    def test_loads_list(self, tmp_path):
        data = [1, 2, 3]
        fpath = tmp_path / "test.pkl"
        with open(fpath, "wb") as f:
            pickle.dump(data, f)
        with open(fpath, "rb") as f:
            loaded = load_pickle(f)
        assert loaded == data

    def test_loads_numpy_array(self, tmp_path):
        data = {"arr": np.array([1.0, 2.0, 3.0])}
        fpath = tmp_path / "test.pkl"
        with open(fpath, "wb") as f:
            pickle.dump(data, f)
        with open(fpath, "rb") as f:
            loaded = load_pickle(f)
        np.testing.assert_array_equal(loaded["arr"], data["arr"])


class TestImread:
    def test_reads_png(self, tmp_path):
        from PIL import Image
        img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        fpath = tmp_path / "test.png"
        img.save(str(fpath))
        result = imread(str(fpath))
        assert result.shape == (10, 10, 3)
        assert result.dtype == np.uint8

    def test_reads_grayscale(self, tmp_path):
        from PIL import Image
        img = Image.fromarray(np.zeros((10, 10), dtype=np.uint8), mode="L")
        fpath = tmp_path / "test.png"
        img.save(str(fpath))
        result = imread(str(fpath))
        assert result.shape == (10, 10)
