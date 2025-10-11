import numpy as np
import matplotlib.pyplot as plt
from utils.helpers import save_json


# ---------------------------------------------------------------------
# 1. Feature statistics
# ---------------------------------------------------------------------
def describe_feature(x):
    """
    Compute basic descriptive statistics for a single feature vector.

    Args:
        x (np.ndarray): 1D array of feature values.

    Returns:
        dict: statistics (mean, median, std, min, max, missing count, missing %)
    """
    x = np.asarray(x)
    mask = ~np.isnan(x)
    n_total = len(x)
    n_valid = np.sum(mask)

    if n_valid == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "missing": n_total,
            "missing_pct": 100.0,
        }

    return {
        "mean": np.mean(x[mask]),
        "median": np.median(x[mask]),
        "std": np.std(x[mask]),
        "min": np.min(x[mask]),
        "max": np.max(x[mask]),
        "missing": n_total - n_valid,
        "missing_pct": 100 * (n_total - n_valid) / n_total,
    }


def feature_skewness(x):
    """Compute skewness (asymmetry) of a 1D feature."""
    x = x[~np.isnan(x)]
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0
    return np.sum(((x - mean) / std) ** 3) / n


def feature_kurtosis(x):
    """Compute kurtosis (tailedness) of a 1D feature."""
    x = x[~np.isnan(x)]
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0
    return np.sum(((x - mean) / std) ** 4) / n - 3


def feature_entropy(x, bins=10):
    """Estimate entropy of a 1D feature (discretized)."""
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return 0
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def detect_duplicate_features(X, tol=1e-10):
    """
    Detect features that are nearly identical (correlation ≈ 1).
    Returns list of tuples (i, j) of duplicates.
    """
    corr = np.corrcoef(X, rowvar=False)
    n = corr.shape[0]
    duplicates = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(corr[i, j] - 1) < tol:
                duplicates.append((i, j))
    return duplicates



def detect_sparse_features(X, threshold=0.95):
    """Return indices of features that are mostly zeros (sparsity > threshold)."""
    sparsity = np.mean(X == 0, axis=0)
    return np.where(sparsity > threshold)[0]


def drop_high_missing_features(X, threshold=0.5):
    """
    Drop features with more than threshold fraction of missing values.
    Returns filtered X and indices of dropped features.
    """
    missing_ratio = np.mean(np.isnan(X), axis=0)
    keep_mask = missing_ratio < threshold
    dropped = np.where(~keep_mask)[0]
    return X[:, keep_mask], dropped

# ---------------------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------------------

def count_missing(X):
    """
    Count missing values (NaNs) per feature.

    Args:
        X (np.ndarray): 2D array

    Returns:
        np.ndarray: counts of missing values per column
    """
    return np.sum(np.isnan(X), axis=0)

def count_missing_unavailable(X, codes=None):
    """
    Count unavailable values per feature (NaN or coded missing values like 7, 9, 77, 99...).

    Args:
        X (np.ndarray): 2D numeric array
        codes (list or None): List of integer codes representing unavailable responses

    Returns:
        np.ndarray: counts of unavailable values per column
    """
    if codes is None:
        codes = [7, 9, 77, 99, 777, 999]

    # Start with NaNs
    missing_mask = np.isnan(X)

    # Add coded-missing mask
    coded_mask = np.isin(X, codes)

    unavailable_mask = missing_mask | coded_mask
    return np.sum(unavailable_mask, axis=0)

# ---------------------------------------------------------------------
# 3. Outlier detection
# ---------------------------------------------------------------------
def find_outliers_iqr(x):
    """
    Identify outliers in a 1D array using the IQR rule.

    Args:
        x (np.ndarray): 1D array of numeric values

    Returns:
        np.ndarray: boolean mask (True = outlier)
    """
    x = x[~np.isnan(x)]
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (x < lower) | (x > upper)

def zscore_outliers(x, threshold=3):
    x = x[~np.isnan(x)]
    z = (x - np.mean(x)) / np.std(x)
    return np.abs(z) > threshold

def quantile_clip(x, lower=0.01, upper=0.99):
    "Quantile clipping to remove extreme values"
    q_low, q_high = np.nanquantile(x, [lower, upper])
    return np.clip(x, q_low, q_high)

def count_outliers_dataset(X, method="iqr"):
    outlier_counts = []
    for j in range(X.shape[1]):
        x = X[:, j]
        if method == "iqr":
            outlier_counts.append(np.sum(find_outliers_iqr(x)))
        elif method == "zscore":
            outlier_counts.append(np.sum(zscore_outliers(x)))
    return np.array(outlier_counts)


# ---------------------------------------------------------------------
# Dataset describe
# ---------------------------------------------------------------------

def dataset_describe(
    X,
    feature_names=None,
    max_features=10,
    save_dir=None,
    save_name="dataset_summary",
    y=None,
    extended=False,
):
    """
    Provide a detailed summary of each feature in the dataset.
    Default: lightweight (basic statistics).
    Extended=True: adds skewness, kurtosis, entropy, variance, outlier %, target correlation, etc.

    Parameters
    ----------
    X : np.ndarray
        Dataset array (n_samples, n_features)
    feature_names : list or None
        Optional list of feature names
    max_features : int
        Number of features to preview in printout
    save_dir : str or None
        Directory to save JSON summary
    save_name : str
        JSON filename (without extension)
    y : np.ndarray or None
        Optional target for correlation analysis
    extended : bool
        Whether to compute advanced diagnostics (default=False)

    Returns
    -------
    list of dicts: detailed summary per feature
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape
    summary = []

    if feature_names is None:
        feature_names = [f"feature_{j}" for j in range(n_features)]

    print(f"\nDataset shape: {n_samples:,} samples × {n_features:,} features")
    header = (
        f"{'Idx':>4s} | {'Name':>15s} | {'dtype':>8s} | {'Unique':>8s} | "
        f"{'Miss':>6s} | {'%Miss':>7s} | {'Mean':>10s} | {'Std':>10s} | {'Min':>10s} | {'Max':>10s}"
    )
    print(header)
    print("-" * len(header))

    possible_missing_codes = [7, 9, 77, 99, 777, 999]

    for j in range(n_features):
        col = X[:, j]
        name = feature_names[j]
        dtype = str(col.dtype)
        mask_valid = ~np.isnan(col)
        n_missing = np.sum(~mask_valid)
        missing_pct = 100 * n_missing / n_samples
        unique_vals = np.unique(col[mask_valid])
        n_unique = len(unique_vals)
        unique_sample = unique_vals[:5].tolist() if n_unique > 0 else []

        # dtype-based type
        if np.issubdtype(col.dtype, np.floating):
            inferred_type = "float"
        elif np.issubdtype(col.dtype, np.integer):
            inferred_type = "integer"
        elif np.issubdtype(col.dtype, np.str_):
            inferred_type = "string"
        else:
            inferred_type = str(col.dtype)

        # Flag potential coded missing values
        has_coded_missing = any(v in possible_missing_codes for v in unique_vals)

        # Numeric stats
        if np.issubdtype(col.dtype, np.number) and np.any(mask_valid):
            mean = np.mean(col[mask_valid])
            std = np.std(col[mask_valid])
            min_val = np.min(col[mask_valid])
            max_val = np.max(col[mask_valid])
            variance = np.var(col[mask_valid])
        else:
            mean = std = min_val = max_val = variance = np.nan

        # Extended diagnostics
        skewness = kurtosis = entropy = outlier_pct = corr = np.nan
        if extended and np.issubdtype(col.dtype, np.number) and np.any(mask_valid):
            skewness = feature_skewness(col)
            kurtosis = feature_kurtosis(col)
            entropy = feature_entropy(col)
            try:
                outlier_mask = find_outliers_iqr(col)
                outlier_pct = 100 * np.sum(outlier_mask) / np.sum(~np.isnan(col))
            except Exception:
                outlier_pct = np.nan

            if y is not None:
                valid_mask = ~np.isnan(col)
                if np.std(col[valid_mask]) > 0 and np.std(y[valid_mask]) > 0:
                    corr = np.corrcoef(col[valid_mask], y[valid_mask])[0, 1]
                else:
                    corr = np.nan


        entry = {
            "index": j,
            "name": name,
            "dtype": dtype,
            "type_inferred": inferred_type,
            "n_unique": int(n_unique),
            "unique_values_sample": unique_sample,
            "missing_count": int(n_missing),
            "missing_pct": float(missing_pct),
            "mean": float(mean),
            "std": float(std),
            "min": float(min_val),
            "max": float(max_val),
            "variance": float(variance),
            "has_coded_missing": bool(has_coded_missing),
        }

        if extended:
            entry.update({
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "entropy": float(entropy),
                "outlier_pct": float(outlier_pct),
                "corr_with_target": float(corr),
            })

        summary.append(entry)

        # Print preview
        if j < max_features:
            print(f"{j:4d} | {name[:15]:15s} | {inferred_type:>8s} | "
                  f"{n_unique:8d} | {n_missing:6d} | {missing_pct:7.2f} | "
                  f"{mean:10.3g} | {std:10.3g} | {min_val:10.3g} | {max_val:10.3g}")
        elif j == max_features:
            print("... (truncated preview)")

    save_json(summary, save_name, save_dir=save_dir)
    print(f"\nSaved dataset summary ({'extended' if extended else 'basic'}) for {n_features} features.\n")

    return summary


def summarize_dataset(X, y=None):
    print(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    print(f"Total missing values: {np.isnan(X).sum()}")
    if y is not None:
        unique, counts = np.unique(y, return_counts=True)
        print(f"Target balance: {dict(zip(unique, counts))}")


# ---------------------------------------------------------------------
# 4. Correlation
# ---------------------------------------------------------------------
def correlation_matrix(X):
    """
    Compute the correlation matrix between all features.

    Args:
        X (np.ndarray): 2D array (n_samples, n_features)

    Returns:
        np.ndarray: correlation matrix (n_features x n_features)
    """
    # Ensure no constant columns (std=0)
    valid_mask = np.std(X, axis=0) > 0
    X_valid = X[:, valid_mask]
    return np.corrcoef(X_valid, rowvar=False)



# ---------------------------------------------------------------------
# 7. Feature transformations
# ---------------------------------------------------------------------

def normalize_feature(x):
    """Min-max normalization of a 1D array."""
    x = x.copy()
    mask = ~np.isnan(x)
    if np.all(~mask):
        return x
    min_x, max_x = np.min(x[mask]), np.max(x[mask])
    if min_x == max_x:
        return np.zeros_like(x)
    x[mask] = (x[mask] - min_x) / (max_x - min_x)
    return x


def standardize_feature(x):
    """Zero-mean, unit-variance scaling of a 1D array."""
    x = x.copy()
    mask = ~np.isnan(x)
    mean, std = np.mean(x[mask]), np.std(x[mask])
    if std == 0:
        return x - mean
    x[mask] = (x[mask] - mean) / std
    return x


def replace_missing_with_mean(x):
    """Replace NaN values with the mean of non-missing values."""
    mask = ~np.isnan(x)
    if np.any(mask):
        mean_val = np.mean(x[mask])
        x[~mask] = mean_val
    return x


# ---------------------------------------------------------------------
# 8. Relationship analysis
# ---------------------------------------------------------------------

def feature_target_correlation(X, y):
    """
    Compute correlation between each feature and the target.
    Works for binary targets (values -1/1 or 0/1).
    """
    corr = []
    for j in range(X.shape[1]):
        valid = ~np.isnan(X[:, j])
        if np.std(X[valid, j]) == 0:
            corr.append(np.nan)
        else:
            corr.append(np.corrcoef(X[valid, j], y[valid])[0, 1])
    return np.array(corr)


def compute_covariance_matrix(X):
    """Compute covariance matrix (centered)."""
    X_centered = X - np.nanmean(X, axis=0)
    X_centered = np.nan_to_num(X_centered)
    n = X_centered.shape[0]
    return (X_centered.T @ X_centered) / (n - 1)


def compute_variance_ratio(X):
    """Compute relative variance contribution per feature."""
    variances = np.nanvar(X, axis=0)
    total = np.nansum(variances)
    return variances / total if total != 0 else np.zeros_like(variances)

