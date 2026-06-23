import os
import numpy as np

SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")


def get_model_path(fname):
    """Return the full file path for a saved model."""
    return os.path.join(SAVED_DIR, fname)


def save_params(fname, params):
    """Save model parameters to disk."""
    fpath = get_model_path(fname)
    np.save(fpath, params)
    print(fname, "saved.")


def load_params(fname):
    """Load model parameters from disk.

    Returns the loaded parameters dict, or None if the file does not exist.
    """
    fpath = get_model_path(fname)
    if not os.path.exists(fpath):
        print(fname, "not available.")
        return None
    params = np.load(fpath, allow_pickle=True).item()
    print(fname, "loaded.")
    return params


def compute_softmax_probs(scores):
    """Numerically stable softmax probabilities.

    Inputs:
    - scores: array of shape (N, C)

    Returns:
    - probs: array of shape (N, C) with softmax probabilities
    """
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)
