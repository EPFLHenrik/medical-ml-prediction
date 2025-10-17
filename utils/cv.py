import numpy as np
import sys, os
sys.path.append(os.path.join(os.getcwd(), '..'))
from implementations import (
    logistic_regression,
    reg_logistic_regression,
    weighted_logistic_regression,
)
from utils.sampling_methods import undersample, oversample, smote  
from utils.all_functions import sigmoid  


def predict(tx, w, threshold=0.5, return_prob=False):
    prob = sigmoid(tx @ w)
    if return_prob:
        return prob
    return (prob >= threshold).astype(int)


def compute_training_error(y, tx, w, threshold=0.5, returning="error"):
    """
    Compute classification metrics for weighted logistic regression (NumPy-only).

    Parameters
    ----------
    y : np.ndarray
        True binary labels (0 or 1)
    tx : np.ndarray
        Feature matrix
    w : np.ndarray
        Trained weights
    threshold : float, optional
        Decision threshold for classification
    returning : str, optional
        Which metric to return. Options:
        "error", "accuracy", "precision", "recall", "f1", "roc_auc", or "all"

    Returns
    -------
    float or dict
        The requested metric value or a dict of all metrics.
    """
    # Predicted probabilities and binary labels
    y = y.flatten()
    proba = sigmoid(tx @ w).flatten()
    y_pred = (proba >= threshold).astype(int)

    # Confusion matrix components
    tp = np.sum((y_pred == 1) & (y == 1))
    tn = np.sum((y_pred == 0) & (y == 0))
    fp = np.sum((y_pred == 1) & (y == 0))
    fn = np.sum((y_pred == 0) & (y == 1))

    # Basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    error = 1 - accuracy
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    roc_auc = compute_roc_auc(y, proba)
    # Return
    metrics = {
        "error": error,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

    if returning == "all":
        return metrics
    elif returning in metrics:
        return metrics[returning]
    else:
        raise ValueError(f"Unknown metric '{returning}'")
    

def evaluate_thresholds(y, tx, w, thresholds=None):
    """
    Evaluate performance metrics (accuracy, error, precision, recall, F1, AUC)
    at multiple thresholds using compute_training_error().
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)
    
    results = []
    
    for t in thresholds:
        metrics = compute_training_error(y, tx, w, threshold=t, returning="all")
        
        results.append({
            "threshold": t,
            "accuracy": metrics["accuracy"],
            "error": metrics["error"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"]
        })
    
    return results


def compute_roc_auc(y, proba):
    # Sort by descending probability
    desc_idx = np.argsort(-proba)
    y_sorted = y[desc_idx]
    proba_sorted = proba[desc_idx]

    P = np.sum(y_sorted == 1)
    N = np.sum(y_sorted == 0)

    tpr = [0.0]
    fpr = [0.0]

    thresholds = np.unique(proba_sorted)

    for thr in thresholds:
        y_pred = (proba_sorted >= thr).astype(int)
        tp = np.sum((y_pred == 1) & (y_sorted == 1))
        fp = np.sum((y_pred == 1) & (y_sorted == 0))
        tpr.append(tp / (P + 1e-10))
        fpr.append(fp / (N + 1e-10))

    tpr.append(1.0)
    fpr.append(1.0)

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    auc = np.trapz(tpr, fpr)
    return auc


def train_logreg(y, tx, kind="logistic",
                 gamma=0.01, lambda_=0.0, max_iters=1000,
                 tol=None, w_init=None, verbose=True):
    """
    Unified training wrapper for logistic regression variants:
    - Standard logistic regression
    - Regularized logistic regression
    - Weighted logistic regression
    """
    n_features = tx.shape[1]
    if w_init is None:
        w_init = np.zeros((n_features, 1))
    y = y.reshape(-1, 1)


    if kind == "logistic":
        # Standard logistic regression (5 args)
        w, losses = logistic_regression(y, tx, w_init, max_iters, gamma)

    elif kind == "reg_logistic":
        # Regularized logistic regression (6 args)
        w, losses = reg_logistic_regression(y, tx, lambda_, w_init, max_iters, gamma)

    elif kind == "w_logistic":
        # Weighted logistic regression (7 args)
        w, losses = weighted_logistic_regression(y, tx, w_init, max_iters, gamma, lambda_, tol)

    else:
        raise ValueError("kind must be 'logistic', 'reg_logistic', or 'w_logistic'")

    final_loss = losses[-1] if isinstance(losses, (list, np.ndarray)) and losses.ndim > 0 else float(losses)

    if verbose:
        print(f"[{kind}] Training finished — final loss: {losses[-1]:.6f}")

    return w, final_loss


def cross_validate(
    model_fn,
    X,
    y,
    k_folds=5,
    seed=None,
    sampling_fn=None,
    sampling_kwargs=None,
    model_kwargs=None,
    predict_fn=None,
    metric_fn=None,
    use_predictions=False,
    verbose=True,
):
    """
    General-purpose k-fold cross-validation with optional sampling and flexible argument control.
    (No inspect library used.)

    Args:
        model_fn (callable): Function or class used to train a model.
                             Should return trained model or weights.
        X (np.ndarray): Feature matrix (n_samples, n_features)
        y (np.ndarray): Label vector (n_samples,)
        k_folds (int): Number of folds for cross-validation.
        seed (int, optional): Random seed for reproducibility.
        sampling_fn (callable, optional): Sampling function (undersample, oversample, smote...).
        sampling_kwargs (dict, optional): Arguments for the sampling function.
        model_kwargs (dict, optional): Arguments for the model training function.
        predict_fn (callable, optional): Function to generate predictions.
                                         Must return (y_pred, y_proba) for logistic-like models.
        metric_fn (callable, optional): Function computing metrics.
        use_predictions (bool): If True, metric_fn expects (y_true, y_pred, y_proba)
                                instead of (y_true, X_val, model).
        verbose (bool): Whether to print per-fold metrics.

    Returns:
        dict: Averaged metrics across folds.
    """
    # Set RNG safely 
    rng = np.random.default_rng(seed)

    y = y.reshape(-1, 1)
    n = len(y)

    # shuffle indices, create folds
    indices = rng.permutation(n)
    folds = np.array_split(indices, k_folds)
    all_metrics = []

    for i, val_idx in enumerate(folds):

        train_idx = np.setdiff1d(indices, val_idx)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # optional sampling on training data only
        if sampling_fn is not None:
            kwargs = sampling_kwargs or {}
            try:
                X_train, y_train = sampling_fn(X_train, y_train, **kwargs)
            except Exception as e:
                print(f"[Warning] Sampling failed in fold {i+1}: {e}")
                continue

        # train modelc (y, tx order!!)
        model_args = model_kwargs or {}
        model = model_fn(y_train, X_train, **model_args)

        # predict
        if predict_fn is None:
            raise ValueError("predict_fn must be provided for evaluation.")
        
        w = model[0] if isinstance(model, tuple) else model
        y_pred, y_proba = predict_fn(X_val, w)

        # evaluate
        if metric_fn is None:
            raise ValueError("metric_fn must be provided for evaluation.")

        if use_predictions:
            metrics = metric_fn(y_val, y_pred, y_proba)
        else:
            metrics = metric_fn(y_val, X_val, w, returning="all")

        all_metrics.append(metrics)

        # print progress
        if verbose:
            msg = f"Fold {i+1}/{k_folds} → " + ", ".join([f"{k}:{v:.3f}" for k, v in metrics.items()])
            print(msg)

    # Compute mean metrics 
    if not all_metrics:
        raise ValueError("No successful folds. Check your sampling or training function.")

    avg_metrics = {m: np.mean([fold[m] for fold in all_metrics]) for m in all_metrics[0]}
    return avg_metrics

