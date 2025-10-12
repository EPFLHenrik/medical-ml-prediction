import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, json, csv
from utils.eda_helpers import count_missing, count_missing_unavailable


# ----- MISSING VALS -------

def plot_missing_values(X, feature_names=None, top_n=None, save_dir=None, file_name=None):
    """
    Plot percentage of missing values per feature.
    Automatically saves under save_dir/file_name if provided.
    """
    n = X.shape[0]
    missing_counts = count_missing(X)
    missing_pct = 100 * missing_counts / n

    # Sort features by missing %
    sorted_idx = np.argsort(missing_pct)[::-1]
    if top_n is not None:
        sorted_idx = sorted_idx[:top_n]

    x = np.arange(len(sorted_idx))
    plt.figure(figsize=(12, 5))
    plt.bar(x, missing_pct[sorted_idx], color="salmon", edgecolor="black")

    if feature_names is not None:
        plt.xticks(x, [feature_names[i] for i in sorted_idx], rotation=90, fontsize=8)
    else:
        plt.xticks(x, sorted_idx, rotation=90, fontsize=8)

    plt.ylabel("Missing values (%)")
    plt.title(f"Missing Values per Feature ({'Top '+str(top_n) if top_n else 'All'})")
    plt.tight_layout()

    if save_dir and file_name:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    plt.show()


def plot_missing_unavailable(X, feature_names=None, top_n=None, codes=None, save_dir=None, file_name=None):
    """
    Plot percentage of unavailable values (NaN or coded) per feature.
    Automatically saves under save_dir/file_name if provided.
    """
    if codes is None:
        codes = [7, 9, 77, 99, 777, 999]

    n = X.shape[0]
    unavailable_counts = count_missing_unavailable(X, codes=codes)
    unavailable_pct = 100 * unavailable_counts / n

    # Sort features by unavailable %
    sorted_idx = np.argsort(unavailable_pct)[::-1]
    if top_n is not None:
        sorted_idx = sorted_idx[:top_n]

    x = np.arange(len(sorted_idx))
    plt.figure(figsize=(12, 5))
    plt.bar(x, unavailable_pct[sorted_idx], color="lightcoral", edgecolor="black")

    if feature_names is not None:
        plt.xticks(x, [feature_names[i] for i in sorted_idx], rotation=90, fontsize=8)
    else:
        plt.xticks(x, sorted_idx, rotation=90, fontsize=8)

    plt.ylabel("Unavailable values (%)")
    plt.title(f"Unavailable (Missing + Coded) Values per Feature ({'Top '+str(top_n) if top_n else 'All'})")
    plt.tight_layout()

    if save_dir and file_name:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    plt.show()


# ----- HISTOGRAMS ----------

def plot_feature_distribution(X, feature_names, idx, y=None, save_dir=None, file_name=None, bins=40, plot = True):
    """
    Plot histogram (and by target if y provided) for one feature.
    Draws target=1 and target=-1 side by side for clarity.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    x = X[:, idx]
    name = feature_names[idx]
    plt.figure(figsize=(6, 4))

    if y is not None:
        x1 = x[y == 1]
        x2 = x[y == -1]
        hist_range = (np.nanmin(x), np.nanmax(x))

        counts1, bins_edges = np.histogram(x1, bins=bins, range=hist_range)
        counts2, _ = np.histogram(x2, bins=bins, range=hist_range)
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
        width = (bins_edges[1] - bins_edges[0]) * 0.4

        plt.bar(bin_centers - width / 2, counts1, width=width,
                color="dodgerblue", edgecolor="black", label="Target = 1")
        plt.bar(bin_centers + width / 2, counts2, width=width,
                color="darkorange", edgecolor="black", label="Target = -1")

        plt.legend(frameon=False)
    else:
        plt.hist(x, bins=bins, color="skyblue", edgecolor="black")

    plt.xlabel(name)
    plt.ylabel("Count")
    plt.title(f"Distribution of {name}")
    plt.tight_layout()

    if save_dir and file_name:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if plot:
        plt.show()
    else:
        plt.close()


def plot_feature_kde_distribution(X, feature_names, idx, y=None, save_dir=None, file_name=None, bins=40, plot=True):
    """
    Plot Seaborn histogram (with KDE overlay) for one feature.
    - If y is provided: plots separate KDE histograms for each class.
    - If plot=False: saves silently without displaying (for batch runs).

    Args:
        X (np.ndarray): feature matrix
        feature_names (list): list of feature names
        idx (int): feature index
        y (np.ndarray or None): optional target array (values -1, 1)
        save_dir (str or None): directory to save plots
        file_name (str or None): name of output file
        bins (int): number of bins for histogram
        plot (bool): whether to display plot (default=True)
    """

    x = X[:, idx]
    name = feature_names[idx]

    # clean nan
    x = x[~np.isnan(x)]
    plt.figure(figsize=(6, 4))

    # Plot
    if y is not None:
        # Prepare valid mask
        valid_mask = ~np.isnan(X[:, idx])
        y_valid = y[valid_mask]
        x_valid = X[valid_mask, idx]

        # Separate classes
        data = {
            "value": x_valid,
            "target": np.where(y_valid == 1, "Target = 1", "Target = -1")
        }
        sns.histplot(
            data=data,
            x="value",
            hue="target",
            bins=bins,
            kde=True,
            element="step",
            stat="count",
            common_norm=False,
            palette={"Target = 1": "dodgerblue", "Target = -1": "darkorange"},
            alpha=0.6,
        )
        plt.legend(title=None, frameon=False)
    else:
        sns.histplot(x, bins=bins, kde=True, color="cornflowerblue", edgecolor=None)

    plt.xlabel(name)
    plt.ylabel("Count")
    plt.title(f"Distribution + KDE of {name}")
    plt.tight_layout()

    # Save
    if save_dir and file_name:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show or close
    if plot:
        plt.show()
    else:
        plt.close()

# ------ BOXPLOTS ------------
def plot_feature_boxplot(X, feature_names, idx, y=None, save_dir=None, file_name=None, plot=True):
    """
    Plot boxplot (optionally by target) for one feature.
    - If y is provided: plots side-by-side boxes for target=1 and -1.
    - If plot=False: saves silently without displaying.

    Args:
        X (np.ndarray): feature matrix
        feature_names (list): feature names
        idx (int): feature index
        y (np.ndarray or None): target vector (values -1, 1)
        save_dir (str or None): directory to save plots
        file_name (str or None): filename for saving
        plot (bool): whether to display or close the plot
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")

    x = X[:, idx]
    name = feature_names[idx]
    valid_mask = ~np.isnan(x)
    x_valid = x[valid_mask]
    y_valid = y[valid_mask] if y is not None else None

    plt.figure(figsize=(4, 4))

    if y is not None:
        data = {
            "value": x_valid,
            "target": np.where(y_valid == 1, "Target = 1", "Target = -1")
        }
        sns.boxplot(
            x="target",
            y="value",
            hue="target",                 
            data=data,
            palette={"Target = 1": "dodgerblue", "Target = -1": "darkorange"},
            width=0.6,
            legend=False,                
        )
        plt.xlabel("Target class")
        plt.title(f"Boxplot of {name} by Target")
    else:
        pastel_blue = "#AEC6CF"  # soft, consistent pastel tone
        sns.boxplot(y=x_valid, color=pastel_blue, width=0.4)
        plt.title(f"Boxplot of {name}")

    plt.ylabel(name)
    plt.tight_layout()

    # Save
    if save_dir and file_name:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if plot:
        plt.show()
    else:
        plt.close()


# ----- ELSE ----------

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
