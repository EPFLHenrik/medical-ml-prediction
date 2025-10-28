import numpy as np
import matplotlib.pyplot as plt
import os
import csv


#basic statistics
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

    # generate default names if none are provided
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

        # infer type
        if n_unique == 1:
            ftype = "constant"
        elif n_unique == 2:
            ftype = "binary"
        else:
            ftype = "continuous"

        # compute stats safely
        if np.any(mask):
            mean = np.mean(col[mask])
            std = np.std(col[mask])
            min_val = np.min(col[mask])
            max_val = np.max(col[mask])
        else:
            mean = std = min_val = max_val = np.nan

        summary.append(
            {
                "index": j,
                "name": feature_names[j],
                "type": ftype,
                "n_unique": n_unique,
                "missing": n_missing,
                "missing_pct": missing_pct,
                "mean": mean,
                "std": std,
                "min": min_val,
                "max": max_val,
            }
        )

        # clean, aligned printout
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

        summary.append(
            {
                "feature_idx": j,
                "type": ftype,
                "n_unique": n_unique,
                "missing": n_missing,
            }
        )

        if j + 1 == max_features:
            print("... (truncated for readability)")
            break

    return summary


# missing values
def count_missing(X):
    """
    Count missing values (NaNs) per feature.

    Args:
        X (np.ndarray): 2D array

    Returns:
        np.ndarray: counts of missing values per column
    """
    return np.sum(np.isnan(X), axis=0)


# outlier detection
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


# correlation
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


# feature level statistics


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


# data diagnostics


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


# feature transformations


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


# relatedness between features and target


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


# visualizations


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
    plt.boxplot(
        x[mask], vert=True, patch_artist=True, boxprops=dict(facecolor="lightgray")
    )
    plt.scatter(
        np.ones(np.sum(outliers)), x[mask][outliers], color="red", label="Outliers"
    )
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

def read_csv_numeric(path): 
    # 1) read header for the name→index mapping 
    with open(path, newline='', encoding='utf-8') as f: 
        reader = csv.reader(f)
        header = next(reader)
    col_idx = {name: i for i, name in enumerate(header)}

    # 2) load the rest as a 2D float array (missing -> NaN)
    # reopen so the file pointer is at the top again
    data = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=1,
        dtype=float,
        missing_values=["", "NA", "NaN"],
        filling_values=np.nan
    )
    # If the CSV has only one data row, genfromtxt returns 1D; force 2D
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return col_idx, data

def clean_data(x_data, x_indices):
    # 6-digit: DK/Ref codes like 777777 / 999999
    six_digit = [
        'FLSHTMY2',
        'HIVTSTD3',
    ]

    # 5-digit: DK/Ref codes like 99999 (e.g., WTKG3)
    five_digit = [
        'PA1VIGM_',
        'PAFREQ2_',
        'WTKG3',
    ]

    # 4-digit: DK/Ref codes like 7777 / 9999 (e.g., WEIGHT2)
    four_digit = [
        'BEANDAY_',
        'FRUTDA1_',
        'FTJUDA1_',
        'GRENDAY_',
        'HEIGHT3',
        'ORNGDAY_',
        'VEGEDA1_',
        'WEIGHT2',
        '_BMI5',
        '_BMI5CAT',
    ]

    # 3-digit: DK/Ref codes like 777 / 999 (e.g., ALCDAY5)
    triple_digit = [
        'ALCDAY5',
        'ASACTLIM',
        'ASNOSLEP',
        'BLDSUGAR',
        'CDHELP',
        'EXERHMM1',
        'EXERHMM2',
        'EXEROFT1',
        'EXEROFT2',
        'FEETCHK2',
        'FRUIT1',
        'FRUITJU1',
        'FVBEANS',
        'FVGREEN',
        'FVORANG',
        'LONGWTCH',
        'STRENGTH',
        'VEGETAB1',
        '_RFBING5',
        'MAXVO2_','FC60_',
    ]

    # 2-digit: DK/Ref codes like 77 / 99 (e.g., DRNK3GE5)
    double_digit = [
        'ADDOWN','ADEAT1','ADENERGY','ADFAIL','ADMOVE','ADPLEASR','ADSLEEP','ADTHINK',
        'ARTTODAY','ASINHALR','ASRCHKUP','ASTHMAGE','CDDISCUS','CDSOCIAL','CHKHEMO3','CRGVEXPT',
        'CVDINFR4','DIABAGE2','DOCTDIAB','EXRACT11',
        'EXRACT21','FEETCHK','HAREHAB1','HHADULT','HOWLONG','HPVADVC2','IMFVPLAC','INCOME2','JOINPAIN',
        'LASTSMK2','LENGEXAM','LSTBLDS3','MENTHLTH','PAFREQ1_','PHYSHLTH','POORHLTH',
        'PSATIME','RDUCHART','SCNTLWK1','SCNTWRK1','VINOCRE2','WHRTST10',
        '_CRACE1','_DRDXAR1','_PRACE1','_RACE_G1','_WT2RAKE','AVEDRNK2',
        'DRNK3GE5','MAXDRNKS','ARTHSOCL','STRFREQ_','CHILDREN',
    ]

    # 1-digit: DK/Ref codes like 7 / 9 (e.g., EXERANY2)
    single_digit = [
        'ADDEPEV2','ARTHDIS2','ARTHEDU','ARTHEXER','ARTHWGT','ASATTACK','ASDRVIST','ASERVIST','ASPUNSAF','ASTHMA3','ASTHMED3','ASYMPTOM',
        'BLDSTOOL','BLIND','BLOODCHO','BPHIGH4','CASTHNO2','CDASSIST','CDHOUSE','CHCCOPD1','CHCKIDNY','CHECKUP1','CHOLCHK','CIMEMLOS',
        'CPDEMO1','CRGVHRS1','CRGVMST2','DECIDE','DIABEDU','DIABEYE','DIFFALON','DRADVISE','EDUCA','EXERANY2','EYEEXAM','FLUSHOT6',
        'HADHYST2','HADMAM','HADPAP2','HADSGCO1','HADSIGM3','HAVARTH3','HIVTST6','HLTHPLN1','HPVADSHT','HPVTEST','LANDLINE','LMTJOIN3',
        'MEDCOST','MISTMNT','NUMHHOL2','PCPSAAD2','PCPSADE1','PCPSADI1','PCPSARE1','PDIABTST','PNEUVAC3','PREDIAB1','PROFEXAM','QLACTLM2',
        'RENTHOM1','SCNTLPAD','SCNTMEL1','SCNTMNY1','SCNTPAID','SEATBELT','SMOKE100','STOPSMK2','STREHAB1','TETANUS','TOLDHI2',
        'USEEQUIP','USENOW3','VICTRCT4','VIDFCLT2','VIEYEXM2','VIGLUMA2','VIMACDG2','VIPRFVS2','VIREDIF3','WTCHSALT','_CHOLCHK','_HCVU651',
        '_LLCPWT','_LMTACT1','_MICHD','_PA150R2','_PA30021','_PA300R2','_PASTRNG','_RACEG21','_RFCHOL','_RFDRHV5','_RFHYPE5','_RFSEAT2',
        '_SMOKER3','_TOTINDA','_VEGESUM','_RFSEAT3','PAMISS1_','GENHLTH','PERSDOC2','BPMEDS','CVDSTRK3','ASTHNOW','CHCOCNCR','CHCSCNCR',
        'DIABETE3','MARITAL','VETERAN3','EMPLOY1','INTERNET','PREGNANT','DIFFWALK','DIFFDRES','CAREGIV1',
        'CRGVHOUS','CRGVPERS','CRGVPRB1','CRGVREL1','LASTSIG3','SXORIENT','TRNSGNDR','RCSGENDR','RCSRLTN2','CASTHDX2',
        'EMTSUPRT','LSATISFY','ADANXEV','_CHISPNC','_RFHLTH','_ASTHMS1','_CASTHM1','_LTASTH1','_MRACE1','_RACEGR3',
        '_RACE','_RFBMI5','_CHLDCNT','_EDUCAG','_INCOMG','_RFSMOK3','_FRTLT1','_VEGLT1',
        'DRNKANY5','_PACAT1','_PAINDX1','_PAREC1','_PASTAE1','_LMTWRK1','_PNEUMO2',
        '_LMTSCL1','_FLSHOT6','_AIDTST3',
    ]

    # no explicit DK/Ref "7…/9…" code in that variable’s own table
    no_dk_rf = [
        'ACTIN11_','ACTIN21_','CADULT','CCLGHOUS','CELLFON2','CELLFON3',
        'COLGHOUS','CRGVLNG1','CSTATE','CTELENUM','CTELNUM1','CVDASPRN','CVDCRHD4',
        'DISPCODE','DROCDY3_','EXACTOT1','EXACTOT2','EXRACT11','EXRACT21','FMONTH',
        'HPLSTTST','HTIN4','HTM4','IDATE','IDAY','IMONTH','INSULIN','IYEAR','LADULT',
        'LASTPAP2','METVL11_','METVL21_','MSCODE','NUMADULT','NUMMEN','NUMPHON2','NUMWOMEN',
        'PA1MIN_','PADUR1_','PADUR2_','PAINACT2','PAMIN11_','PAMIN21_','PAVIG11_','PAVIG21_','PCDMDECN',
        'PCPSARS1','PSATEST1','PVTRESD1','PVTRESD2','QLHLTH2','QLMENTL2','QLSTRES2','QSTLANG','QSTVER',
        'RDUCSTRK','RLIVPAIN','SEQNO','SEX','SHINGLE2','SMOKDAY2','STATERES','VIINSUR2',
        '_AGE65YR','_AGE80','_AGEG5YR','_AGE_G','_CLLCPWT','_CPRACE',
        '_DRNKWEK','_DUALCOR','_DUALUSE','_FRT16','_FRTRESP','_FRUITEX','_FRUTSUM','_HISPANC',
        '_MINAC11','_MINAC21','_MISFRTN','_MISVEGN',
        '_PSU','_RAWRAKE','_STATE','_STRWT',
        '_STSTR','_VEG23','_VEGETEX','_VEGRESP',
    ]

    x_data[:, [x_indices["STRFREQ_"]] == 99, x_indices["STRFREQ_"]] = np.nan
    x_data[:, [x_indices["DIABAGE2"]] == 98, x_indices["DIABAGE2"]] = np.nan
    x_data[:, [x_indices["LASTSMK2"]] == 8, x_indices["LASTSMK2"]] = 0
    x_data[:, [x_indices["EXRACT11"]] == 8, x_indices["EXRACT11"]] = 0
    x_data[:, [x_indices["SCNTWRK1"]] == 98, x_indices["SCNTWRK1"]] = 0
    x_data[:, [x_indices["DIABAGE2"]] == 98, x_indices["DIABAGE2"]] = 0
    x_data[:, [x_indices["SCNTWRK1"]] == 97, x_indices["SCNTWRK1"]] = np.nan
    x_data[:, [x_indices["SCNTLWK1"]] == 98, x_indices["SCNTLWK1"]] = 0
    x_data[:, [x_indices["SCNTLWK1"]] == 97, x_indices["SCNTLWK1"]] = np.nan
    x_data[:, [x_indices["DROCDY3_"]] == 900, x_indices["DROCDY3_"]] = np.nan
    x_data[:, [x_indices["_DRNKWEK"]] == 99900, x_indices["_DRNKWEK"]] = np.nan
    x_data[:, [x_indices["PAFREQ1_"]] == 99900, x_indices["PAFREQ1_"]] = np.nan
    x_data[:, [x_indices["PAFREQ2_"]] == 99900, x_indices["PAFREQ2_"]] = np.nan
    x_data[:, [x_indices["PAFREQ1_"]] == 90, x_indices["PAFREQ1_"]] = np.nan
    x_data[:, [x_indices["PAFREQ2_"]] == 99, x_indices["PAFREQ2_"]] = np.nan
    
    
    
    for c in x_indices:
        if c in single_digit:
            x_data[x_data[:, x_indices[c]] == 7, x_indices[c]] = np.nan
            x_data[x_data[:, x_indices[c]] == 8, x_indices[c]] = 0
            x_data[x_data[:, x_indices[c]] == 9, x_indices[c]] = np.nan
        elif c in double_digit:
            x_data[x_data[:, x_indices[c]] == 77, x_indices[c]] = np.nan
            x_data[x_data[:, x_indices[c]] == 88, x_indices[c]] = 0
            x_data[x_data[:, x_indices[c]] == 99, x_indices[c]] = np.nan
        elif c in triple_digit:
            x_data[x_data[:, x_indices[c]] == 777, x_indices[c]] = np.nan
            x_data[x_data[:, x_indices[c]] == 555, x_indices[c]] = 0
            x_data[x_data[:, x_indices[c]] == 888, x_indices[c]] = 0
            x_data[x_data[:, x_indices[c]] == 999, x_indices[c]] = np.nan
        elif c in four_digit:
            x_data[x_data[:, x_indices[c]] == 7777, x_indices[c]] = np.nan
            x_data[x_data[:, x_indices[c]] == 8888, x_indices[c]] = 0
            x_data[x_data[:, x_indices[c]] == 9999, x_indices[c]] = np.nan
        elif c in five_digit:
            x_data[x_data[:, x_indices[c]] == 77777, x_indices[c]] = np.nan
            x_data[x_data[:, x_indices[c]] == 88888, x_indices[c]] = 0
            x_data[x_data[:, x_indices[c]] == 99999, x_indices[c]] = np.nan
        elif c in six_digit:
            x_data[x_data[:, x_indices[c]] == 777777, x_indices[c]] = np.nan
            x_data[x_data[:, x_indices[c]] == 888888, x_indices[c]] = 0
            x_data[x_data[:, x_indices[c]] == 999999, x_indices[c]] = np.nan
        else:
            if c not in no_dk_rf: 
                raise Exception("Not in any group", c)
        
    x_data[x_data[:, x_indices["WEIGHT2"]] <= 1000, x_indices["WEIGHT2"]] = x_data[x_data[:, x_indices["WEIGHT2"]] <= 1000, x_indices["WEIGHT2"]] * 0.453592
    x_data[x_data[:, x_indices["WEIGHT2"]] >= 9000, x_indices["WEIGHT2"]] = x_data[x_data[:, x_indices["WEIGHT2"]] >= 9000, x_indices["WEIGHT2"]] - 9000
    x_data[x_data[:, x_indices["HEIGHT3"]] <= 1000, x_indices["HEIGHT3"]] = ((x_data[x_data[:, x_indices["HEIGHT3"]] <= 1000, x_indices["HEIGHT3"]]) // 100) * 30.48 + ((x_data[x_data[:, x_indices["HEIGHT3"]] <= 1000, x_indices["HEIGHT3"]]) % 100) * 2.54
    x_data[x_data[:, x_indices["HEIGHT3"]] >= 9000, x_indices["HEIGHT3"]] = x_data[x_data[:, x_indices["HEIGHT3"]] >= 9000, x_indices["HEIGHT3"]] - 9000
    x_data[x_data[:, x_indices["WEIGHT2"]] > 200, x_indices["WEIGHT2"]] = np.nan

    return x_data
