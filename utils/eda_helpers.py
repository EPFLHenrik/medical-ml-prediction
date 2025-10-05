import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Basic statistics
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


def dataset_describe(X, feature_names=None, max_features=10):
    """
    Provide a detailed summary of each feature in the dataset:
    - feature name and index
    - type (constant, binary, continuous)
    - unique count
    - missing count and %
    - mean, std, min, max (for numeric features)

    Args:
        X (np.ndarray): Dataset array (n_samples, n_features)
        feature_names (list or None): Optional list of feature names
        max_features (int): Maximum number of features to display

    Returns:
        list of dicts: detailed summary per feature
    """
    n_samples, n_features = X.shape
    summary = []

    # Generate default names if none are provided
    if feature_names is None:
        feature_names = [f"feature_{j}" for j in range(n_features)]

    print(f"\nDataset shape: {n_samples:,} samples × {n_features:,} features")
    header = (
        f"{'Idx':>4s} | {'Name':>15s} | {'Type':>10s} | {'Unique':>7s} | "
        f"{'Missing':>8s} | {'%Miss':>6s} | "
        f"{'Mean':>12s} | {'Std':>12s} | {'Min':>12s} | {'Max':>12s}"
    )
    print(header)
    print("-" * len(header))

    for j in range(n_features):
        col = X[:, j]
        mask = ~np.isnan(col)
        n_missing = np.sum(~mask)
        n_unique = len(np.unique(col[mask]))
        missing_pct = 100 * n_missing / n_samples

        # Infer type
        if n_unique == 1:
            ftype = "constant"
        elif n_unique == 2:
            ftype = "binary"
        else:
            ftype = "continuous"

        # Compute stats safely
        if np.any(mask):
            mean = np.mean(col[mask])
            std = np.std(col[mask])
            min_val = np.min(col[mask])
            max_val = np.max(col[mask])
        else:
            mean = std = min_val = max_val = np.nan

        summary.append({
            "index": j,
            "name": feature_names[j],
            "type": ftype,
            "n_unique": n_unique,
            "missing": n_missing,
            "missing_pct": missing_pct,
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val
        })

        # Clean, aligned printout
        print(
            f"{j:4d} | {feature_names[j][:15]:15s} | {ftype:10s} | "
            f"{n_unique:7d} | {n_missing:8d} | {missing_pct:6.2f} | "
            f"{mean:12.3g} | {std:12.3g} | {min_val:12.3g} | {max_val:12.3g}"
        )

        if j + 1 == max_features:
            print("... (truncated for readability)")
            break

    return summary



def summarize_dataset(X, y=None):
    print(f"Samples: {X.shape[0]} | Features: {X.shape[1]}")
    print(f"Total missing values: {np.isnan(X).sum()}")
    print(f"Constant features: {len(detect_constant_features(X))}")
    print(f"Binary features: {len(detect_binary_features(X))}")
    if y is not None:
        unique, counts = np.unique(y, return_counts=True)
        print(f"Target balance: {dict(zip(unique, counts))}")


def inspect_feature_types(X, max_features=10):
    """
    Inspect each feature to infer its data type (constant, binary, continuous),
    number of unique values, and count of missing entries.

    Args:
        X (np.ndarray): Dataset array of shape (n_samples, n_features)
        max_features (int): Maximum number of features to print (default=10)
    
    Returns:
        list of dicts: summary info for each feature
    """
    n_features = X.shape[1]
    summary = []

    print(f"\nInspecting {n_features} features:")
    print(f"{'Feature':>8s} | {'Type':>10s} | {'Unique':>8s} | {'Missing':>8s}")
    print("-" * 45)

    for j in range(n_features):
        col = X[:, j]
        n_missing = np.isnan(col).sum()
        unique_vals = np.unique(col[~np.isnan(col)])
        n_unique = len(unique_vals)

        # infer feature type roughly
        if n_unique == 1:
            ftype = "constant"
        elif n_unique == 2:
            ftype = "binary"
        else:
            ftype = "continuous"

        print(f"{j:8d} | {ftype:10s} | {n_unique:8d} | {n_missing:8d}")

        summary.append({
            "feature_idx": j,
            "type": ftype,
            "n_unique": n_unique,
            "missing": n_missing
        })

        if j + 1 == max_features:
            print("... (truncated for readability)")
            break

    return summary



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
# 5. Feature-level statistics
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# 6. Data quality diagnostics
# ---------------------------------------------------------------------

def detect_constant_features(X):
    """Return indices of features with zero variance."""
    return np.where(np.std(X, axis=0) == 0)[0]


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


def detect_binary_features(X):
    """Return indices of features containing only two unique values."""
    binary_features = []
    for i in range(X.shape[1]):
        vals = np.unique(X[:, i][~np.isnan(X[:, i])])
        if len(vals) == 2:
            binary_features.append(i)
    return np.array(binary_features)


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


# ---------------------------------------------------------------------
# 9. Visualization utilities
# ---------------------------------------------------------------------

def plot_feature_vs_target(X, y, feature_idx):
    """Visualize feature distribution by target class."""
    feature = X[:, feature_idx]
    mask1 = y == 1
    mask0 = y == -1

    plt.hist(feature[mask1], bins=30, alpha=0.6, label="Class 1", color="dodgerblue")
    plt.hist(feature[mask0], bins=30, alpha=0.6, label="Class -1", color="salmon")
    plt.xlabel(f"Feature {feature_idx}")
    plt.ylabel("Count")
    plt.title(f"Feature {feature_idx} distribution by class")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_feature_pairs(X, features):
    """Pairwise scatterplots for a few features."""
    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(X[:, features[i]], bins=30, color="gray")
            else:
                ax.scatter(X[:, features[j]], X[:, features[i]], s=5, alpha=0.5)
            if i == n - 1:
                ax.set_xlabel(f"f{features[j]}")
            if j == 0:
                ax.set_ylabel(f"f{features[i]}")
    plt.tight_layout()
    plt.show()


def plot_target_balance(y):
    """Plot distribution of target classes (for classification)."""
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(unique.astype(str), counts, color=["salmon", "dodgerblue"])
    plt.xlabel("Target class")
    plt.ylabel("Count")
    plt.title("Target Class Balance")
    plt.grid(alpha=0.3)
    plt.show()


def plot_outlier_distribution(x):
    """Boxplot showing outliers explicitly."""
    mask = ~np.isnan(x)
    outliers = np.abs(x[mask] - np.mean(x[mask])) > 3 * np.std(x[mask])
    plt.boxplot(x[mask], vert=True, patch_artist=True, boxprops=dict(facecolor="lightgray"))
    plt.scatter(np.ones(np.sum(outliers)), x[mask][outliers], color="red", label="Outliers")
    plt.title("Outlier Visualization")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_feature_importance(values, names=None, top_k=15):
    """Visualize top correlated or most variant features."""
    idx = np.argsort(np.abs(values))[::-1][:top_k]
    vals = values[idx]
    labels = [f"f{i}" for i in idx] if names is None else np.array(names)[idx]

    plt.barh(labels, vals, color="steelblue")
    plt.xlabel("Importance value")
    plt.title("Top Feature Importances")
    plt.gca().invert_yaxis()
    plt.show()


def plot_histogram(x, bins=30, title=None):
    """Plot histogram of a feature."""
    x = x[~np.isnan(x)]
    plt.hist(x, bins=bins, color="steelblue", edgecolor="black")
    plt.title(title or "Feature distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.show()


def plot_box(x, title=None):
    """Plot boxplot of a feature."""
    x = x[~np.isnan(x)]
    plt.boxplot(x, vert=True, patch_artist=True, boxprops=dict(facecolor="lightgray"))
    plt.title(title or "Feature boxplot")
    plt.ylabel("Value")
    plt.grid(alpha=0.3)
    plt.show()


def plot_missing_values(X):
    """
    Plot bar chart of missing values per feature.

    Args:
        X (np.ndarray): dataset array
    """
    n_missing = count_missing(X)
    plt.bar(np.arange(len(n_missing)), n_missing, color="salmon", edgecolor="black")
    plt.xlabel("Feature index")
    plt.ylabel("Missing values")
    plt.title("Missing values per feature")
    plt.show()


def plot_correlation_heatmap(corr):
    """
    Plot correlation matrix heatmap.

    Args:
        corr (np.ndarray): correlation matrix
    """
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation coefficient")
    plt.title("Feature Correlation Heatmap")
    plt.xlabel("Feature index")
    plt.ylabel("Feature index")
    plt.show()
