"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):
    col_mean = x.mean(axis=0)
    col_std = x.std(axis=0)

    x = np.where(col_std != 0, (x - col_mean) / col_std, 0)
    return x

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    return {"w": np.zeros((n_features, )), "b": 0}

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    w = params["w"]
    b = params["b"]
    
    return x @ w + b

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    return np.where(scores >= 0, 1, -1)

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    return max(0, 1 - y*score)

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    # W = params["w"]

    # scores = compute_scores(x, params)

    # np_hinge_loss = np.vectorize(hinge_loss_example)
    # loss = np_hinge_loss(scores, y)
    # mean_loss = np.mean(loss)
    
    # return mean_loss + reg_lambda*(W @ W.T)
    scores = compute_scores(x, params)

    loss = np.maximum(0, 1 - y * scores)
    mean_loss = np.mean(loss)

    return mean_loss + reg_lambda * np.sum(params["w"] ** 2)

# Step 7 - compute_gradients
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    """Return {'dw': ndarray shape (n_features,), 'db': float} = gradient of svm_objective."""
    scores = compute_scores(x, params)

    margin = 1 - y * scores
    mask = margin > 0

    dw = -((x * y[:, None]) * mask[:, None]).mean(axis=0)
    dw += 2 * reg_lambda * params["w"]

    db = -(y * mask).mean()

    return {"dw": dw, "db": db}

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    new_w = params["w"] - learning_rate*grads["dw"]
    new_b = params["b"] - learning_rate*grads["db"]

    return {"w": new_w, "b": new_b}

# Step 9 - train_svm (not yet solved)
# TODO: implement

# Step 10 - predict_labels (not yet solved)
# TODO: implement

# Step 11 - accuracy_score (not yet solved)
# TODO: implement

