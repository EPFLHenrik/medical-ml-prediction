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
    thresholds=None,
    use_predictions=False,
    verbose=True,
):
    """
    General-purpose k-fold cross-validation with optional sampling,
    flexible argument control, and multi-threshold evaluation.

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
        thresholds (float or list, optional): Threshold(s) for prediction evaluation.
        use_predictions (bool): If True, metric_fn expects (y_true, y_pred, y_proba)
                                instead of (y_true, X_val, model).
        verbose (bool): Whether to print per-fold metrics.

    Returns:
        dict: Averaged metrics across folds. (or across folds × thresholds)
    """
    rng = np.random.default_rng(seed)
    y = y.reshape(-1, 1)
    n = len(y)

    # shuffle indices,  create folds
    indices = rng.permutation(n)
    folds = np.array_split(indices, k_folds)
    all_metrics = []

    # ensure thresholds is iterable
    if thresholds is not None and not isinstance(thresholds, (list, np.ndarray)):
        thresholds = [thresholds]

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

        # train model (note y, X order)
        model_args = model_kwargs or {}
        model = model_fn(y_train, X_train, **model_args)
        w = model[0] if isinstance(model, tuple) else model

        if predict_fn is None:
            raise ValueError("predict_fn must be provided for evaluation.")
        if metric_fn is None:
            raise ValueError("metric_fn must be provided for evaluation.")

        # evaluate at one or multiple thresholds
        if thresholds is not None:
            for thr in thresholds:
                y_pred, y_proba = predict_fn(X_val, w)
                metrics = metric_fn(y_val, X_val, w, threshold=thr, returning="all")
                metrics["threshold"] = thr
                all_metrics.append(metrics)
                if verbose:
                    print(f"Fold {i+1}/{k_folds} @ thr={thr:.2f} → " +
                          ", ".join([f"{k}:{v:.3f}" for k, v in metrics.items() if k != "threshold"]))
        else:
            y_pred, y_proba = predict_fn(X_val, w)
            metrics = metric_fn(y_val, X_val, w, returning="all")
            all_metrics.append(metrics)
            if verbose:
                msg = f"Fold {i+1}/{k_folds} → " + ", ".join([f"{k}:{v:.3f}" for k, v in metrics.items()])
                print(msg)

    if not all_metrics:
        raise ValueError("No successful folds. Check your sampling or training function.")

    # avg metrics
    if thresholds is not None:
        avg_metrics = {}
        for t in thresholds:
            subset = [m for m in all_metrics if abs(m["threshold"] - t) < 1e-9]
            avg_metrics[t] = {k: np.mean([m[k] for m in subset])
                              for k in subset[0] if k not in ["threshold"]}
        return avg_metrics
    else:
        avg_metrics = {m: np.mean([fold[m] for fold in all_metrics]) for m in all_metrics[0]}
        return avg_metrics


def analyze_cv_results(weights_list, train_metrics, val_metrics, class_weights_list=None, verbose=True):
    """
    Analyze cross-validation diagnostics for over/underfitting and weight stability (NumPy-only).

    Parameters
    ----------
    weights_list : list of np.ndarray
        Learned weight vectors from each fold. Shape per fold: (n_features, 1)
    train_metrics : list of dict
        Metrics computed on training sets (each with keys: f1, accuracy, etc.)
    val_metrics : list of dict
        Metrics computed on validation sets
    class_weights_list : list of np.ndarray or list of dict, optional
        If provided, contains class weights used per fold.
    verbose : bool, optional
        If True, prints summary diagnostics.

    Returns
    -------
    dict
        Summary with numeric diagnostics.
    """
    n_folds = len(weights_list)
    weights = np.hstack([w.reshape(-1, 1) for w in weights_list]).T  # (n_folds, n_features)

    # ---- 1) Compute mean/std for metrics ----
    train_f1s = np.array([m["f1"] for m in train_metrics])
    val_f1s   = np.array([m["f1"] for m in val_metrics])
    train_mean, val_mean = train_f1s.mean(), val_f1s.mean()
    train_std, val_std = train_f1s.std(ddof=1), val_f1s.std(ddof=1)
    gap = train_mean - val_mean

    # ---- 2) Coefficient diagnostics ----
    mean_w = weights.mean(axis=0)
    std_w = weights.std(axis=0, ddof=1)
    rel_var = std_w / (np.abs(mean_w) + 1e-12)
    coef_norms = np.linalg.norm(weights, axis=1)
    mean_norm, std_norm = coef_norms.mean(), coef_norms.std(ddof=1)

    # Pairwise distance (average distance between folds)
    pairwise = np.sqrt(((weights[:, None, :] - weights[None, :, :]) ** 2).sum(axis=2))
    mean_pairwise = np.mean(pairwise)
    max_pairwise = np.max(pairwise)

    # Sign flip count
    mean_sign = np.sign(mean_w)
    sign_flips = np.sum(np.sign(weights) != mean_sign, axis=0)
    unstable_feats = np.argsort(-rel_var)[:5]  # top 5 unstable

    # ---- 3) Class weight variability (if any) ----
    if class_weights_list is not None and len(class_weights_list) > 1:
        cw_arr = np.array([
            list(cw.values()) if isinstance(cw, dict) else np.array(cw).flatten()
            for cw in class_weights_list
        ])
        cw_mean = cw_arr.mean(axis=0)
        cw_std = cw_arr.std(axis=0, ddof=1)
        rel_cw_change = cw_std / (np.abs(cw_mean) + 1e-12)
    else:
        cw_mean = cw_std = rel_cw_change = None

    # ---- 4) Diagnose fit quality ----
    if gap > 0.08 and train_mean >= 0.8:
        verdict = "OVERFITTING"
    elif train_mean < 0.65 and val_mean < 0.65:
        verdict = "UNDERFITTING"
    elif val_std > 0.05:
        verdict = "HIGH VARIANCE (unstable folds)"
    else:
        verdict = "OK / WELL FITTED"

    summary = {
        "train_mean": train_mean,
        "val_mean": val_mean,
        "train_std": train_std,
        "val_std": val_std,
        "gap": gap,
        "verdict": verdict,
        "mean_coef_norm": mean_norm,
        "std_coef_norm": std_norm,
        "mean_pairwise_dist": mean_pairwise,
        "max_pairwise_dist": max_pairwise,
        "unstable_features_idx": unstable_feats,
        "top_rel_var": rel_var[unstable_feats],
        "sign_flips_top": sign_flips[unstable_feats],
        "coef_variability_mean": rel_var.mean(),
    }

    if cw_mean is not None:
        summary["class_weight_mean"] = cw_mean
        summary["class_weight_rel_change"] = rel_cw_change

    if verbose:
        print("\n=== Cross-validation Diagnostics ===")
        print(f"Train F1 mean: {train_mean:.3f} ± {train_std:.3f}")
        print(f"Val   F1 mean: {val_mean:.3f} ± {val_std:.3f}")
        print(f"Gap (train - val): {gap:.3f} → {verdict}")
        print(f"Weight L2 norm mean: {mean_norm:.3f} ± {std_norm:.3f}")
        print(f"Mean pairwise distance between folds: {mean_pairwise:.3f}")
        print(f"Top 5 unstable features (by relative var): {unstable_feats.tolist()}")
        if cw_mean is not None:
            print(f"Class weight relative change: {rel_cw_change}")
        print("====================================\n")

    return summary
