import os
from contextlib import redirect_stdout


def print_repo_tree(
    max_depth=3, exclude=("venv_ml", "__pycache__", ".git", ".gitignore")
):
    """
    Print the folder structure of the repository hierarchically,
    starting from the git root if available.
    """
    # Find git repo root (if .git folder exists somewhere above)
    current = os.getcwd()
    while current != os.path.dirname(current):  # stop at filesystem root
        if ".git" in os.listdir(current):
            repo_root = current
            break
        current = os.path.dirname(current)
    else:
        repo_root = os.getcwd()  # fallback

    print(f"Repository root: {repo_root}\n")

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in exclude]
        level = root.replace(repo_root, "").count(os.sep)
        if level > max_depth:
            continue
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root) or '.'}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            if f not in exclude:
                print(f"{subindent}{f}")


def save_repo_tree(
    filename="repo_structure.txt",
    max_depth=3,
    exclude=("venv_ml", "__pycache__", ".git", ".gitignore"),
):
    """
    Save the folder structure of the repository hierarchically into a text file.
    """
    # Detect the repo root
    current = os.getcwd()
    while current != os.path.dirname(current):
        if ".git" in os.listdir(current):
            repo_root = current
            break
        current = os.path.dirname(current)
    else:
        repo_root = os.getcwd()

    filepath = os.path.join(repo_root, filename)

    # Print and save at the same time
    with open(filepath, "w", encoding="utf-8") as f, redirect_stdout(f):
        print(f"Repository root: {repo_root}\n")
        for root, dirs, files in os.walk(repo_root):
            dirs[:] = [d for d in dirs if d not in exclude]
            level = root.replace(repo_root, "").count(os.sep)
            if level > max_depth:
                continue
            indent = " " * 4 * level
            print(f"{indent}{os.path.basename(root) or '.'}/")
            subindent = " " * 4 * (level + 1)
            for file in files:
                if file not in exclude:
                    print(f"{subindent}{file}")

    print(f"\n✅ Repository structure saved to: {filepath}")


# Print it
print_repo_tree(max_depth=3)

# Run it
save_repo_tree("repo_structure.txt", max_depth=3)
