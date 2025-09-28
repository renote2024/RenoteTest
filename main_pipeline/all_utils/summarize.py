
import argparse
import ast
import json
import os
from unittest import result
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde


def sort_failed_modules(df, post_fix):
    module_counter = Counter()

    # Iterate through each row's failed_modules
    for modules_str in df[f"failed_modules_{post_fix}"]:
        if pd.notna(modules_str) and modules_str.strip():
            try:
                modules = ast.literal_eval(modules_str)
                module_counter.update(modules)
            except Exception as e:
                print(f"Skipping invalid failed_modules entry: {modules_str} ({e})")

    # Sort by frequency (high to low)
    sorted_modules = module_counter.most_common()

    return sorted_modules


def format_comparison_results(df):
    results = []
    results.append("\nTest Results Comparison:\n")
    results.append(f"{'Metric':<26} | {'Pre':>25} | {'Post':>25} | {'Change':>25}")
    results.append(f"{'':<26} | {'(% total / % self)':>25} | {'(% total / % self)':>25} | {'(% total / % self)':>25}")
    results.append("-" * 110)

    total_tests_pre = df['total_test_pre'].sum()
    total_tests_post = df['total_test_post'].sum()

    def fmt_dual_pct(value, total_base, self_base):
        pct_total = (value / total_base * 100) if total_base else 0
        pct_self = (value / self_base * 100) if self_base else 0
        return f"{value:>4} ({pct_total:5.1f}% / {pct_self:5.1f}%)"

    def fmt_single_pct(value, base):
        pct = (value / base * 100) if base else 0
        return f"{value:>4} ({pct:5.1f}%)"

    def fmt_change_dual(pre, post, total_base, self_base):
        diff = post - pre
        pct_total_change = (diff / total_base * 100) if total_base else 0
        pct_self_change = (diff / self_base * 100) if self_base else 0
        return f"{diff:+5} ({pct_total_change:+6.1f}% / {pct_self_change:+6.1f}%)"

    def fmt_change_single(pre, post, total_base):
        diff = post - pre
        pct = (diff / total_base * 100) if total_base else 0
        return f"{diff:+5} ({pct:+6.1f}%)"

    # Linear totals (passed + failed)
    linear_success_pre = df[df['linear_pre'] == 'success'].shape[0]
    linear_success_post = df[df['linear_post'] == 'success'].shape[0]
    linear_fail_pre = df[df['linear_pre'] == 'failed'].shape[0]
    linear_fail_post = df[df['linear_post'] == 'failed'].shape[0]
    linear_total_pre = linear_success_pre + linear_fail_pre
    linear_total_post = linear_success_post + linear_fail_post

    metrics = [
        ("Total tests", total_tests_pre, total_tests_post, (total_tests_pre, total_tests_post)),
        ("Passed tests", df['success_test_pre'].sum(), df['success_test_post'].sum(), (total_tests_pre, total_tests_post)),
        ("Failed tests", df['failed_test_pre'].sum(), df['failed_test_post'].sum(), (total_tests_pre, total_tests_post)),
    ]

    linear_present = (linear_total_pre > 0) or (linear_total_post > 0)
    if linear_present:
        metrics += [
            ("Passed linear tests", linear_success_pre, linear_success_post, (total_tests_pre, linear_total_post)),
            ("Failed linear tests", linear_fail_pre, linear_fail_post, (total_tests_pre, linear_total_post)),
        ]

    for label, pre, post, (total_base, self_base) in metrics:
        if label in ["Passed linear tests", "Failed linear tests"]:
            pre_fmt = fmt_dual_pct(pre, total_base, self_base)
            post_fmt = fmt_dual_pct(post, total_base, self_base)
            change_fmt = fmt_change_dual(pre, post, total_base, self_base)
        else:
            pre_fmt = fmt_single_pct(pre, total_base)
            post_fmt = fmt_single_pct(post, self_base)
            change_fmt = fmt_change_single(pre, post, total_base)
        results.append(f"{label:<26} | {pre_fmt:>25} | {post_fmt:>25} | {change_fmt:>25}")

    return results


def format_failed_modules(df):
    results = []
    results.append("\nNotebook Failure Highlights:\n")
    results.append(f"{'Metric':<20} | {'Pre (Notebook – Fails)':<28} | {'Post (Notebook – Fails)':<28}")
    results.append(f"{'-'*20}-+-{'-'*28}-+-{'-'*28}")

    failed_modules_pre = sort_failed_modules(df, 'pre')
    failed_modules_post = sort_failed_modules(df, 'post')
    highest_failed_pre = failed_modules_pre[0] if failed_modules_pre else ("-", 0)
    highest_failed_post = failed_modules_post[0] if failed_modules_post else ("-", 0)
    lowest_failed_pre = failed_modules_pre[-1] if failed_modules_pre else ("-", 0)
    lowest_failed_post = failed_modules_post[-1] if failed_modules_post else ("-", 0)

    if failed_modules_pre:
        pre_high = f"{highest_failed_pre[0]} – {highest_failed_pre[1]}"
        post_high = f"{highest_failed_post[0]} – {highest_failed_post[1]}"
        pre_low = f"{lowest_failed_pre[0]} – {lowest_failed_pre[1]}"
        post_low = f"{lowest_failed_post[0]} – {lowest_failed_post[1]}"

        results.append(f"{'Highest Failures':<20} | {pre_high:<28} | {post_high:<28}")
        results.append(f"{'Lowest Failures':<20} | {pre_low:<28} | {post_low:<28}")
    else:
        results.append(f"{'No failed modules':<20} | {'-':<28} | {'-':<28}")

    return results

def cumulative_passed_results(df):
    df["before_fix"] = df["success_test_pre"].cumsum()
    df["after_fix"] = df["success_test_post"].cumsum()

    # Add notebook counter (1-based index)
    df["notebook"] = range(1, len(df) + 1)

    # Select only the needed columns
    output_df = df[["notebook", "before_fix", "after_fix"]]
    return output_df

def normalize_passed_tests(df):
    df["before_percent"] = (df["success_test_pre"] / df["total_test_pre"]) * 100
    df["after_percent"] = (df["success_test_post"] / df["total_test_post"]) * 100
    df["notebook"] = range(1, len(df) + 1)

    output_df = df[["notebook", "before_percent", "after_percent"]].copy()

    # Sort each column independently
    output_df["before_percent"] = np.sort(output_df["before_percent"].values)
    output_df["after_percent"] = np.sort(output_df["after_percent"].values)

    return output_df

def cumulative_failed_results(df):
    df["before_fix"] = df["failed_test_pre"].cumsum()
    df["after_fix"] = df["failed_test_post"].cumsum()

    # Add notebook counter (1-based index)
    df["notebook"] = range(1, len(df) + 1)

    # Select only the needed columns
    output_df = df[["notebook", "before_fix", "after_fix"]]
    return output_df

def normalize_failed_tests(df):
    df["before_percent"] = (df["failed_test_pre"] / df["total_test_pre"]) * 100
    df["after_percent"] = (df["failed_test_post"] / df["total_test_post"]) * 100
    df["notebook"] = range(1, len(df) + 1)

    output_df = df[["notebook", "before_percent", "after_percent"]].copy()

    # Sort each column independently
    output_df["before_percent"] = np.sort(output_df["before_percent"].values)
    output_df["after_percent"] = np.sort(output_df["after_percent"].values)

    return output_df

def cumulative_passed_results_executable(df):
    df["executable_before_fix"] = df["success_test_pre"].where(df['linear_pre'] == 'success', 0).cumsum()
    df["executable_after_fix"]  = df["success_test_post"].where(df['linear_post'] == 'success', 0).cumsum()
    df["non_executable_before_fix"] = df["success_test_pre"].where(df['linear_pre'] == 'failed', 0).cumsum()
    df["non_executable_after_fix"]  = df["success_test_post"].where(df['linear_post'] == 'failed', 0).cumsum()

    # Add notebook counter (1-based index)
    df["notebook"] = range(1, len(df) + 1)

    # Select only the needed columns
    output_df = df[["notebook", "executable_before_fix", "executable_after_fix", "non_executable_before_fix", "non_executable_after_fix"]]
    return output_df

def normalize_cumulative_results_executable(df):
    # Compute cumulative sums for executable and non-executable
    df["executable_before_fix"] = df["success_test_pre"].where(df['linear_pre'] == 'success', 0).cumsum()
    df["executable_after_fix"]  = df["success_test_post"].where(df['linear_post'] == 'success', 0).cumsum()
    df["non_executable_before_fix"] = df["success_test_pre"].where(df['linear_pre'] == 'failed', 0).cumsum()
    df["non_executable_after_fix"]  = df["success_test_post"].where(df['linear_post'] == 'failed', 0).cumsum()

    # Normalize by total cumulative sum
    df["executable_before_percent"] = df["executable_before_fix"] / df["executable_before_fix"].iloc[-1] * 100
    df["executable_after_percent"]  = df["executable_after_fix"] / df["executable_after_fix"].iloc[-1] * 100
    df["non_executable_before_percent"] = df["non_executable_before_fix"] / df["non_executable_before_fix"].iloc[-1] * 100
    df["non_executable_after_percent"]  = df["non_executable_after_fix"] / df["non_executable_after_fix"].iloc[-1] * 100

    # Add notebook counter (1-based index)
    df["notebook"] = range(1, len(df) + 1)

    # Select only the normalized percent columns + notebook
    output_df = df[["notebook",
                    "executable_before_percent", "executable_after_percent",
                    "non_executable_before_percent", "non_executable_after_percent"]].copy()

    # Optionally, sort each column independently like your failed tests function
    output_df["executable_before_percent"] = np.sort(output_df["executable_before_percent"].values)
    output_df["executable_after_percent"]  = np.sort(output_df["executable_after_percent"].values)
    output_df["non_executable_before_percent"] = np.sort(output_df["non_executable_before_percent"].values)
    output_df["non_executable_after_percent"]  = np.sort(output_df["non_executable_after_percent"].values)

    return output_df

def plot_df_line_chart(output_df, filename):
    plt.figure(figsize=(8, 4))
    n = len(output_df)

    # Percent on X-axis, notebook/rank on Y-axis
    plt.scatter(output_df["before_percent"], range(1, n+1), 
                label="Before", marker="o", s=5)
    plt.scatter(output_df["after_percent"], range(1, n+1), 
                label="After", marker="s", s=1)

    plt.xlabel("Percent")
    plt.ylabel("Notebook")
    # plt.title("Before vs After (Sorted Distributions)")
    plt.legend()
    plt.grid(True)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def plot_passed_tests_hist(output_df, filename):
    plt.figure(figsize=(8, 5))

    # Define bins: 0–10%, 10–20%, ..., 90–100%
    bins = np.arange(0, 110, 10)

    # Histogram counts for before and after
    before_counts, _ = np.histogram(output_df["before_percent"], bins=bins)
    after_counts, _ = np.histogram(output_df["after_percent"], bins=bins)

    # Bar positions
    x = np.arange(len(before_counts))  # 10 intervals
    width = 0.4  # bar width

    # Plot side-by-side bars
    plt.bar(x - width/2, before_counts, width=width, label="Before")
    plt.bar(x + width/2, after_counts, width=width, label="After")

    # X-axis labels
    bin_labels = [f"{bins[i]}–{bins[i+1]}%" for i in range(len(bins)-1)]
    plt.xticks(x, bin_labels, rotation=45)

    plt.xlabel("Passed Test Percentage Interval")
    plt.ylabel("Number of Notebooks")
    plt.title("Distribution of Passed Tests (Before vs After)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def plot_passed_tests_density(output_df, output_path, bin_size = 5):
    # plt.figure(figsize=(10, 5))

    # Define bins
    bins = np.arange(0, 100 + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Histogram counts
    before_counts, _ = np.histogram(output_df["before_percent"], bins=bins)
    after_counts, _ = np.histogram(output_df["after_percent"], bins=bins)

    # x = np.arange(len(before_counts))
    # width = 0.4

    # Bars
    # plt.bar(x - width/2, before_counts, width=width, label="Before", color="#1f77b4")
    # plt.bar(x + width/2, after_counts, width=width, label="After", color="#ff7f0e")

    # --- Compute KDE counts per bin ---
    kde_before = gaussian_kde(output_df["before_percent"].dropna())
    kde_after = gaussian_kde(output_df["after_percent"].dropna())
    xs = np.linspace(0, 100, 1000)

    # KDE scaled to counts in each bin
    kde_before_counts = []
    kde_after_counts = []
    for i in range(len(bins)-1):
        mask = (xs >= bins[i]) & (xs < bins[i+1])
        kde_before_counts.append(kde_before(xs[mask]).sum() * (xs[1]-xs[0]) * len(output_df))
        kde_after_counts.append(kde_after(xs[mask]).sum() * (xs[1]-xs[0]) * len(output_df))

    kde_before_counts = np.array(kde_before_counts)
    kde_after_counts = np.array(kde_after_counts)

    # Plot KDE lines
    # plt.plot(x, kde_before_counts, color="#1f77b4", linestyle="--", label="Before (KDE)")
    # plt.plot(x, kde_after_counts, color="#ff7f0e", linestyle="--", label="After (KDE)")

    # --- Normal distribution counts per bin ---
    mu_b, std_b = norm.fit(output_df["before_percent"].dropna())
    mu_a, std_a = norm.fit(output_df["after_percent"].dropna())
    norm_before_counts = []
    norm_after_counts = []
    for i in range(len(bins)-1):
        # approximate count in bin = cdf(b+1) - cdf(b) * total number
        cdf_b = norm.cdf(bins[i+1], mu_b, std_b) - norm.cdf(bins[i], mu_b, std_b)
        cdf_a = norm.cdf(bins[i+1], mu_a, std_a) - norm.cdf(bins[i], mu_a, std_a)
        norm_before_counts.append(cdf_b * len(output_df))
        norm_after_counts.append(cdf_a * len(output_df))

    # Plot normal curves
    # plt.plot(x, norm_before_counts, color="green", alpha=0.4, label=f"Before Normal (μ={mu_b:.1f}, σ={std_b:.1f})")
    # plt.plot(x, norm_after_counts, color="purple", alpha=0.4, label=f"After Normal (μ={mu_a:.1f}, σ={std_a:.1f})")

    np.savetxt(f"{output_path}/bin_centers.csv", bin_centers, delimiter=",")
    np.savetxt(f"{output_path}/before_counts.csv", before_counts, delimiter=",")
    np.savetxt(f"{output_path}/after_counts.csv", after_counts, delimiter=",")
    np.savetxt(f"{output_path}/kde_before.csv", np.column_stack([bin_centers, kde_before_counts]), delimiter=",")
    np.savetxt(f"{output_path}/kde_after.csv", np.column_stack([bin_centers, kde_after_counts]), delimiter=",")
    np.savetxt(f"{output_path}/norm_before_counts.csv", norm_before_counts, delimiter=",")
    np.savetxt(f"{output_path}/norm_after_counts.csv", norm_after_counts, delimiter=",")

    # X-axis labels
    # bin_labels = [f"{int(bins[i])}–{int(bins[i+1])}%" for i in range(len(bins)-1)]
    # plt.xticks(x, bin_labels, rotation=45)

    # plt.xlabel("Passed Test Percentage")
    # plt.ylabel("# of Notebooks")
    # # plt.title(f"Distribution of Passed Tests (Before vs After)")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(filename, bbox_inches="tight")
    # plt.close()

def export_passed_tests_tex(output_df, tex_path, bin_size=5):
    """
    Export histogram, KDE, and normal distribution data into a .tex file
    with \begin{filecontents*} blocks, ready for pgfplots.
    """

    # Define bins
    bins = np.arange(0, 100 + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Histogram counts
    before_counts, _ = np.histogram(output_df["before_percent"].dropna(), bins=bins)
    after_counts, _ = np.histogram(output_df["after_percent"].dropna(), bins=bins)

    # --- KDE counts per bin ---
    kde_before = gaussian_kde(output_df["before_percent"].dropna())
    kde_after = gaussian_kde(output_df["after_percent"].dropna())
    xs = np.linspace(0, 100, 1000)

    kde_before_counts, kde_after_counts = [], []
    for i in range(len(bins) - 1):
        mask = (xs >= bins[i]) & (xs < bins[i+1])
        kde_before_counts.append(kde_before(xs[mask]).sum() * (xs[1]-xs[0]) * len(output_df))
        kde_after_counts.append(kde_after(xs[mask]).sum() * (xs[1]-xs[0]) * len(output_df))

    # --- Normal distribution counts per bin ---
    mu_b, std_b = norm.fit(output_df["before_percent"].dropna())
    mu_a, std_a = norm.fit(output_df["after_percent"].dropna())

    norm_before_counts, norm_after_counts = [], []
    for i in range(len(bins) - 1):
        cdf_b = norm.cdf(bins[i+1], mu_b, std_b) - norm.cdf(bins[i], mu_b, std_b)
        cdf_a = norm.cdf(bins[i+1], mu_a, std_a) - norm.cdf(bins[i], mu_a, std_a)
        norm_before_counts.append(cdf_b * len(output_df))
        norm_after_counts.append(cdf_a * len(output_df))

    # --- Write to TeX file ---
    with open(tex_path, "w") as f:
        def write_filecontents(name, xvals, yvals):
            f.write(f"\\begin{{filecontents*}}{{{name}}}\n")
            f.write("bin,count\n")
            for x, y in zip(xvals, yvals):
                f.write(f"{x},{y}\n")
            f.write("\\end{filecontents*}\n\n")

        # Write datasets
        write_filecontents("before_counts.csv", bin_centers, before_counts)
        write_filecontents("after_counts.csv", bin_centers, after_counts)
        write_filecontents("kde_before.csv", bin_centers, kde_before_counts)
        write_filecontents("kde_after.csv", bin_centers, kde_after_counts)
        write_filecontents("norm_before.csv", bin_centers, norm_before_counts)
        write_filecontents("norm_after.csv", bin_centers, norm_after_counts)

def plot_cumulative_line_chart(output_df, filename):
    plt.figure(figsize=(8, 5))
    n = len(output_df)

    # Scatter plot: percent on X-axis, notebook on Y-axis
    plt.scatter(output_df["executable_before_percent"], range(1, n+1),
                label="Executable Before", marker="o", s=1)
    plt.scatter(output_df["executable_after_percent"], range(1, n+1),
                label="Executable After", marker="s", s=1)
    plt.scatter(output_df["non_executable_before_percent"], range(1, n+1),
                label="Non-Executable Before", marker="^", s=1)
    plt.scatter(output_df["non_executable_after_percent"], range(1, n+1),
                label="Non-Executable After", marker="v", s=1)

    plt.xlabel("Percent")
    plt.ylabel("Notebook / Rank")
    plt.title("Cumulative Passed Results (Normalized)")
    plt.legend()
    plt.grid(True)

    # Save as vector-based PDF
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def main(csv_file_path, output_file_path):
    results = []

    output_dir = os.path.dirname(output_file_path)
    graph_path = os.path.join(output_dir, "graphs")
    plot_path = os.path.join(output_dir, "plots")

    os.makedirs(graph_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    results.append(f"Summary of {csv_file_path}\n")

    # Read CSV file
    df = pd.read_csv(csv_file_path)
    unique_repos = df['repo_path'].nunique()
    total_nbs = len(df)
    improved_nbs = df[df['success_test_pre'] < df['success_test_post']].shape[0]
    worsened_nbs = df[df['success_test_pre'] > df['success_test_post']].shape[0]

    results.append(f"Overall results:")
    results.append(f"- Total unique repositories: {unique_repos}")
    results.append(f"- Total notebooks: {total_nbs}")
    results.append(f"- Total notebooks that have improved: {improved_nbs}")
    results.append(f"- Total notebooks that have worsened: {worsened_nbs}")

    # Cumulative passed results
    passed_df = cumulative_passed_results(df)
    passed_df_downsampled = passed_df.iloc[::10, :]  # Downsample by taking every 10th row
    passed_file_csv = f"{graph_path}/cumulative_passed_results.csv"
    passed_df_downsampled.to_csv(passed_file_csv, index=False)

    # Normalized passed tests
    normalized_passed_df = normalize_passed_tests(df)
    normalized_passed_df_downsampled = normalized_passed_df
    normalized_passed_file_csv = f"{graph_path}/normalized_passed_results.csv"
    normalized_passed_df_downsampled.to_csv(normalized_passed_file_csv, index=False)
    plot_df_line_chart(normalized_passed_df_downsampled, f"{plot_path}/normalized_passed_results.pdf")
    plot_passed_tests_hist(normalized_passed_df_downsampled, f"{plot_path}/normalized_passed_results_hist.pdf")
    plot_passed_tests_density(normalized_passed_df_downsampled, graph_path)
    export_passed_tests_tex(normalized_passed_df_downsampled, f"{graph_path}/normalized_passed_results.tex")

    # Cumulative failed results summary
    failed_df = cumulative_failed_results(df)
    failed_df_downsampled = failed_df.iloc[::10, :]  # Downsample by taking every 10th row
    failed_file_csv = f"{graph_path}/cumulative_failed_results.csv"
    failed_df_downsampled.to_csv(failed_file_csv, index=False)

    # Normalized failed tests
    normalized_failed_df = normalize_failed_tests(df)
    normalized_failed_df_downsampled = normalized_failed_df
    normalized_failed_file_csv = f"{graph_path}/normalized_failed_results.csv"
    normalized_failed_df_downsampled.to_csv(normalized_failed_file_csv, index=False)
    plot_df_line_chart(normalized_failed_df_downsampled, f"{plot_path}/normalized_failed_results.pdf")

    # Cumulative passed tests for executable and non-executable pre and post fix
    exec_df = cumulative_passed_results_executable(df)
    exec_df_downsampled = exec_df.iloc[::10, :]  # Downsample by taking every 10th row
    exec_file_csv = f"{graph_path}/cumulative_passed_results_executable.csv"
    exec_df_downsampled.to_csv(exec_file_csv, index=False)

    # Normalized executable tests
    normalized_exec_df = normalize_cumulative_results_executable(df)
    normalized_exec_df_downsampled = normalized_exec_df
    normalized_exec_file_csv = f"{graph_path}/normalized_cumulative_results_executable.csv"
    normalized_exec_df_downsampled.to_csv(normalized_exec_file_csv, index=False)
    plot_cumulative_line_chart(normalized_exec_df_downsampled, f"{plot_path}/normalized_cumulative_results_executable.pdf")

    results.extend(format_comparison_results(df))
    results.extend(format_failed_modules(df))

    results.append("\nPassed Linear Tests with failed tests:")
    linear_passed_with_failed_pre = df[(df['linear_pre'] == 'success') & (df['failed_test_pre'] > 0)].shape[0]
    linear_passed_with_failed_post = df[(df['linear_post'] == 'success') & (df['failed_test_post'] > 0)].shape[0]

    results.append(f"- Pre: {linear_passed_with_failed_pre}")
    results.append(f"- Post: {linear_passed_with_failed_post}")    
    
    # Save to a txt file
    with open(output_file_path, 'w') as f:
        for result in results:
            f.write(result + "\n")


    # results.append("\nNotebook Failure Highlights:\n")
    # results.append(f"{'Metric':<20} | {'Pre (Notebook – Fails)':<28} | {'Post (Notebook – Fails)':<28}")
    # results.append(f"{'-'*20}-+-{'-'*28}-+-{'-'*28}")

    # failed_modules_pre = sort_failed_modules(df, 'pre')
    # failed_modules_post = sort_failed_modules(df, 'post')
    # highest_failed_pre = failed_modules_pre[0] if failed_modules_pre else ("-", 0)
    # highest_failed_post = failed_modules_post[0] if failed_modules_post else ("-", 0)
    # lowest_failed_pre = failed_modules_pre[-1] if failed_modules_pre else ("-", 0)
    # lowest_failed_post = failed_modules_post[-1] if failed_modules_post else ("-", 0)

    # if failed_modules_pre:
    #     pre_high = f"{highest_failed_pre[0]} – {highest_failed_pre[1]}"
    #     post_high = f"{highest_failed_post[0]} – {highest_failed_post[1]}"
    #     pre_low = f"{lowest_failed_pre[0]} – {lowest_failed_pre[1]}"
    #     post_low = f"{lowest_failed_post[0]} – {lowest_failed_post[1]}"

    #     results.append(f"{'Highest Failures':<20} | {pre_high:<28} | {post_high:<28}")
    #     results.append(f"{'Lowest Failures':<20} | {pre_low:<28} | {post_low:<28}")
    # else:
    #     results.append(f"{'No failed modules':<20} | {'-':<28} | {'-':<28}")


    # results.append("\nTest Results Comparison:\n")
    # results.append(f"{'Metric':<28} | {'Pre':>7} | {'Post':>7} | {'Change':>8} | {'% of Total':>10}")
    # results.append("-" * 75)

    # # Collect values
    # total_test_pre = df['total_test_pre'].sum()
    # total_test_post = df['total_test_post'].sum()
    # failed_test_pre = df['failed_test_pre'].sum()
    # failed_test_post = df['failed_test_post'].sum()
    # passed_test_pre = df['success_test_pre'].sum()
    # passed_test_post = df['success_test_post'].sum()
    # linear_success_pre = df[df['linear_pre'] == 'success'].shape[0]
    # linear_success_post = df[df['linear_post'] == 'success'].shape[0]
    # linear_failed_pre = df[df['linear_pre'] == 'failed'].shape[0]
    # linear_failed_post = df[df['linear_post'] == 'failed'].shape[0]

    # def format_line(label, pre, post, post_base=None):
    #     change = post - pre
    #     percent = f"{(post / post_base) * 100:.2f}%" if post_base else "-"
    #     return f"{label:<28} | {pre:>7} | {post:>7} | {change:>+8} | {percent:>10}"

    # results.append(format_line("Total tests", total_test_pre, total_test_post))
    # results.append(format_line("Passed tests", passed_test_pre, passed_test_post, total_test_post))
    # results.append(format_line("Failed tests", failed_test_pre, failed_test_post, total_test_post))
    # results.append(format_line("Passed linear tests", linear_success_pre, linear_success_post, total_nbs))
    # results.append(format_line("Failed linear tests", linear_failed_pre, linear_failed_post, total_nbs))


    # post_fix = ["pre", "post"]

    # # Total number of tests
    # for post_fix in post_fix:
    #     results.append(f"\nResult of {post_fix}_execution tests")
    #     total_tests = df[f'total_test_{post_fix}'].sum()
    #     results.append(f"- Total number of tests: {total_tests}")

    #     # Total number of passed/failed tests and percentage
    #     total_failed_tests = df[f'failed_test_{post_fix}'].sum()
    #     total_passed_tests = df[f'success_test_{post_fix}'].sum()
    #     results.append(f"- Total number of failed tests: {total_failed_tests}")
    #     results.append(f"   + of total tests: {total_failed_tests} = {(total_failed_tests / total_tests) * 100:.2f}%")
    #     results.append(f"- Total number of passed tests: {total_passed_tests}")
    #     results.append(f"   + of total tests: {total_passed_tests} = {(total_passed_tests / total_tests) * 100:.2f}%")

    #     # Total number of passed/failed linear tests and percentage
    #     total_passed_linear = df[df[f'linear_{post_fix}'] == 'success'].shape[0]
    #     total_failed_linear = df[df[f'linear_{post_fix}'] == 'failed'].shape[0]
    #     results.append(f"- Total number of passed linear tests: {total_passed_linear}")
    #     results.append(f"   + of total tests: {total_passed_linear} = {(total_passed_linear / total_tests) * 100:.2f}%")
    #     results.append(f"   + of total linear tests: {total_passed_linear} = {(total_passed_linear / total_nbs) * 100:.2f}%")
    #     results.append(f"- Total number of failed linear tests: {total_failed_linear}")
    #     results.append(f"   + of total tests: {total_failed_linear} = {(total_failed_linear / total_tests) * 100:.2f}%")
    #     results.append(f"   + of total linear tests: {total_failed_linear} = {(total_failed_linear / total_nbs) * 100:.2f}%")

    #     # Highest/lowest number of failed modules in a notebook
    #     failed_modules = sort_failed_modules(df, post_fix)
    #     if failed_modules:
    #         highest_failed_module = failed_modules[0]
    #         lowest_failed_module = failed_modules[-1]
    #         results.append(f"- Highest number of failed modules: `{highest_failed_module[0]}` - {highest_failed_module[1]} failures")
    #         results.append(f"- Lowest number of failed modules: `{lowest_failed_module[0]}` - {lowest_failed_module[1]} failures")
    #     else:
    #         results.append("- Failed modules found: 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a repository of notebooks.')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file containing repository data.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output file.')
    args = parser.parse_args()

    main(args.csv, args.output)