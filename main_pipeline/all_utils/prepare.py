import pandas as pd
import os
import argparse
from main_pipeline.renote_utils.nb_utils import check_notebook_validity


def check_and_filter_notebooks(all_repos):
    """
    Filter notebooks that are valid (not in ipynb_checkpoints and pass validity check).
    """
    filtered = {}
    for repo_path, notebooks in all_repos.items():
        valid_nbs = []
        for nb_path in notebooks:
            if "ipynb_checkpoints" in nb_path:
                continue
            _, status = check_notebook_validity(nb_path)
            if status == "Success":
                valid_nbs.append(nb_path)
        if valid_nbs:
            filtered[repo_path] = valid_nbs
    return filtered

def prepare_data(root, output_csv):
    repo_data = {}

    # Walk through every folder under root
    for dirpath, dirnames, filenames in os.walk(root):
        # Collect all notebooks in this folder recursively
        nb_paths = [
            os.path.join(dp, f)
            for dp, _, files in os.walk(dirpath)
            for f in files if f.endswith(".ipynb")
        ]

        if nb_paths:
            repo_data[os.path.normpath(dirpath)] = [os.path.normpath(p) for p in nb_paths]
            # Avoid double-counting nested repos
            dirnames[:] = []

    # Filter notebooks before saving
    filtered_repos = check_and_filter_notebooks(repo_data)

    if not filtered_repos:
        print("⚠️ No valid notebooks found after filtering.")
        return

    # Prepare rows for CSV
    rows = []
    for repo_path, nb_paths in filtered_repos.items():
        rows.append({
            "repo_path": repo_path,
            "nb_paths": ";".join(nb_paths)
        })

    df = pd.DataFrame(rows, columns=["repo_path", "nb_paths"])
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df)} repos with valid notebooks → {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for analysis.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    prepare_data(args.root_dir, args.output)