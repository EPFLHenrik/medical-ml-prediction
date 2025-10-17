import numpy as np

def undersample(X, y, ratio=1.0, seed=None):
    """
    Random undersampling of the majority class.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        y (np.ndarray): Label vector of shape (n_samples,)
        ratio (float): Desired ratio of majority:minority samples to keep
                       Example: ratio=2.0 → keep twice as many majority samples as minority ones.
        seed (int, optional): Random seed for reproducibility

    Returns:
        (X_resampled, y_resampled): Tuple of NumPy arrays containing
                                    the undersampled dataset.
    """
    if y.ndim == 2:
        y = y.ravel()

    if seed is not None:
        np.random.seed(seed)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    # identifying majority and minority class
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2:
        raise ValueError("y must contain exactly two classes.")

    minority_label = labels[np.argmin(counts)]
    majority_label = labels[np.argmax(counts)]
    minority_idx = np.where(y == minority_label)[0]
    majority_idx = np.where(y == majority_label)[0]

    n_min = len(minority_idx)
    n_maj_keep = int(n_min * ratio)
    n_maj_keep = min(n_maj_keep, len(majority_idx))

    # randomly sample 
    keep_majority = np.random.choice(majority_idx, size=n_maj_keep, replace=False)

    # combine and shuffle
    keep_idx = np.concatenate([minority_idx, keep_majority])
    np.random.shuffle(keep_idx)

    X_resampled = X[keep_idx]
    y_resampled = y[keep_idx].reshape(-1, 1)

    return X_resampled, y_resampled

def oversample(X, y, ratio=1.0, seed=None):
    """
    Random oversampling of the minority class.

    Duplicates existing minority samples (with replacement) until the
    desired majority:minority ratio is achieved.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Label vector of shape (n_samples,).
        ratio (float): Desired majority:minority ratio.
                       Example: ratio=1.0 → balance classes equally,
                       ratio=2.0 → keep twice as many majority samples as minority.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        (X_resampled, y_resampled): Tuple of NumPy arrays containing
                                    the oversampled dataset.
    """
    if y.ndim == 2:
        y = y.ravel()

    if seed is not None:
        np.random.seed(seed)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2:
        raise ValueError("y must contain exactly two classes.")

    # identify minority and majority
    minority_label = labels[np.argmin(counts)]
    majority_label = labels[np.argmax(counts)]
    minority_idx = np.where(y == minority_label)[0]
    majority_idx = np.where(y == majority_label)[0]

    n_min = len(minority_idx)
    n_maj = len(majority_idx)

    # compute how many minority samples to add
    n_min_target = int(n_maj / ratio)
    n_to_add = max(0, n_min_target - n_min)

    if n_to_add == 0:
        # already balanced or minority larger than target
        return X.copy(), y.copy()

    # randomly sample (with replacement) from minority indices
    add_idx = np.random.choice(minority_idx, size=n_to_add, replace=True)
    X_add = X[add_idx]
    y_add = y[add_idx]

    # combine
    X_resampled = np.vstack([X, X_add])
    y_resampled = np.concatenate([y, y_add])

    # shuffle combined data
    perm = np.random.permutation(len(y_resampled))
    X_resampled = X_resampled[perm]
    y_resampled = y_resampled[perm].reshape(-1, 1)

    return X_resampled, y_resampled

def smote(X, y, k=5, ratio=1.0, seed=None):
    """
    SMOTE (Synthetic Minority Oversampling Technique).

    Generates synthetic minority samples by interpolating between each
    minority instance and one of its k nearest minority neighbors.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Label vector of shape (n_samples,).
        k (int): Number of nearest neighbors among minority samples (default=5).
        ratio (float): Desired majority:minority ratio.
                       Example: ratio=1.0 → balance classes equally.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        (X_resampled, y_resampled): Tuple of NumPy arrays containing
                                    the dataset after SMOTE oversampling.
    """
    if y.ndim == 2:
        y = y.ravel()

    if seed is not None:
        np.random.seed(seed)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2:
        raise ValueError("y must contain exactly two classes.")

    # identify minority and majority
    minority_label = labels[np.argmin(counts)]
    majority_label = labels[np.argmax(counts)]

    X_min = X[y == minority_label]
    X_maj = X[y == majority_label]
    n_min, n_maj = len(X_min), len(X_maj)

    # how many synthetic minority samples are needed
    n_min_target = int(n_maj / ratio)
    n_to_add = max(0, n_min_target - n_min)
    if n_to_add == 0:
        return X.copy(), y.copy()

    # pairwise distances among minority samples
    diff = X_min[:, None, :] - X_min[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    # for each sample, get indices of its k nearest minority neighbors
    np.fill_diagonal(dist, np.inf)
    nn_indices = np.argsort(dist, axis=1)[:, :k]

    # generate synthetic samples
    new_samples = []
    for _ in range(n_to_add):
        i = np.random.randint(0, n_min)
        nn = np.random.choice(nn_indices[i])
        lam = np.random.rand()  # interpolation weight
        x_new = X_min[i] + lam * (X_min[nn] - X_min[i])
        new_samples.append(x_new)

    X_new = np.array(new_samples)
    y_new = np.full(len(X_new), minority_label)

    # combine and shuffle
    X_resampled = np.vstack([X, X_new])
    y_resampled = np.concatenate([y, y_new])
    perm = np.random.permutation(len(y_resampled))

    X_resampled = X_resampled[perm]
    y_resampled = y_resampled[perm].reshape(-1, 1)

    return X_resampled, y_resampled

def smote_then_undersample(X, y, k=5, over_ratio=1.0, under_ratio=1.0, seed=None):
    """
    Hybrid resampling: SMOTE oversampling followed by random undersampling.

    Step 1: Apply SMOTE to increase minority samples.
    Step 2: Randomly undersample majority samples to fine-tune class balance.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Label vector of shape (n_samples,).
        k (int): Number of nearest neighbors for SMOTE (default 5).
        over_ratio (float): Target majority:minority ratio after SMOTE (default 1.0 → balance).
        under_ratio (float): Desired final majority:minority ratio after undersampling
                             (default 1.0 → balance).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        (X_resampled, y_resampled): Tuple of NumPy arrays after hybrid resampling.
    """
    if y.ndim == 2:
        y = y.ravel()
    
    if seed is not None:
        np.random.seed(seed)

    X_sm, y_sm = smote(X, y, k=k, ratio=over_ratio, seed=seed)
    X_res, y_res = undersample(X_sm, y_sm, ratio=under_ratio, seed=seed)
    y_res = y_res.reshape(-1, 1)
    return X_res, y_res

