import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


GROUP_COLORS = {
    "Concatenated Repetition": "#2B8A20",
    "Rotated Surface": "#F4320B",
    "Rotated Surface MSP": "#DBB71D",
    "Rotated Surface Tanner": "#E955C8",
    "Hamming MSP": "#27BAA3",
    "Hamming Tanner": "#278BF5",
    "Holographic": "#695454",
    "BB MSP": "#6E4BDE",
    "BB Tanner": "#3C5DC9",
}

RENAME_MAP = {
    "Rotated Surface MSP": "RSC MSP",
    "Rotated Surface Tanner": "RSC Tanner",
    "Rotated Surface": "RSC",
    "Hamming MSP": "Ham MSP",
    "Hamming Tanner": "Ham Tanner",
    "Concatenated Repetition": "Concat Rep",
}

REP_ORDER = [
    "Concatenated Repetition",
    "Holographic",
    "Rotated Surface",
    "Rotated Surface Tanner",
    "Rotated Surface MSP",
    "Hamming Tanner",
    "Hamming MSP",
    "BB Tanner",
    "BB MSP",
]


def add_brute_force_costs(df):
    mapping = {
        ("Concatenated Repetition", 9): 2 ** (9 - 1),
        ("Concatenated Repetition", 27): 2 ** (27 - 1),
        ("Concatenated Repetition", 81): 2 ** (81 - 1),
        ("Rotated Surface", 9): 2 ** (9 - 1),
        ("Rotated Surface", 25): 2 ** (25 - 1),
        ("Rotated Surface", 49): 2 ** (49 - 1),
        ("Rotated Surface", 81): 2 ** (81 - 1),
        ("Rotated Surface MSP", 9): 2 ** (9 - 1),
        ("Rotated Surface MSP", 25): 2 ** (25 - 1),
        ("Rotated Surface Tanner", 9): 2 ** (9 - 1),
        ("Rotated Surface Tanner", 25): 2 ** (25 - 1),
        ("Rectangular Surface", 9): 2 ** (9 - 1),
        ("Rectangular Surface", 15): 2 ** (15 - 1),
        ("Rectangular Surface", 21): 2 ** (21 - 1),
        ("Hamming MSP", 7): 2 ** (7 - 1),
        ("Hamming MSP", 15): 2 ** (15 - 7),
        ("Hamming Tanner", 7): 2 ** (7 - 1),
        ("Hamming Tanner", 15): 2 ** (15 - 7),
        ("Hamming MSP", 31): 2 ** (31 - 21),
        ("Holographic", 25): 2 ** (25 - 11),
        ("Holographic", 95): 2 ** (95 - 51),
        ("BB MSP", 18): 2 ** (18 - 4),
        ("BB MSP", 30): 2 ** (30 - 4),
        ("BB Tanner", 18): 2 ** (18 - 4),
        ("BB Tanner", 30): 2 ** (30 - 4),
    }
    df["brute_force_cost"] = df.set_index(["tensor_network", "num_qubits"]).index.map(
        mapping
    )
    return df


def plot_log_operations_bar_chart(
    contraction_cost_file, out_file="bar_chart_log_operations.png", method=None
):
    """Plots a bar chart comparing the contraction costs using default dense and custom SST cost functions.

    Args:
        contraction_cost_file: File name that contains data in CSV format.
        out_file: Output file name for the plot.
    """
    df = pd.read_csv(contraction_cost_file, sep=";")
    df = df[
        ~(
            (df["tensor_network"] == "BB MSP")
            & (df["num_qubits"] == 30)
            & (df["methods"] == "['kahypar']")
        )
    ]  # remove BB MSP with 30 qubits, don't have full data

    if method is not None:
        df = df[df["methods"].apply(lambda x: f"{method}" in x)]

    if os.path.exists("results/data/optimal_costs.csv"):
        optimal_df = pd.read_csv("results/data/optimal_costs.csv", sep=";")
        optimal_df = optimal_df[optimal_df["cost_fn"] == "custom_flops"]

    methods = sorted(df["methods"].unique())
    n_methods = len(methods)
    nrows = 1
    ncols = math.ceil(n_methods / nrows)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12 * ncols, 6), sharey=True
    )
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Make a subplot for each method
    for idx, method in enumerate(methods):
        ax = axes[idx]

        df_m = df[df["methods"] == method]
        df_m["operations_log2"] = np.log2(df_m["operations_with_bruteforce"])

        # Need the not log values to compute overall improvement ratios
        grouped_df_not_log = (
            df_m.groupby(["tensor_network", "num_qubits", "cost_fn"])[
                "operations_with_bruteforce"
            ]
            .agg(
                avg_ops="mean",
                std_dev="std",
                min_ops="min",
                max_ops="max",
                num_rows="count",
            )
            .reset_index()
        )

        not_log_df = grouped_df_not_log.pivot_table(
            index=["tensor_network", "num_qubits"], columns="cost_fn", values="avg_ops"
        ).reset_index()

        not_log_df["improvement_default_custom"] = round(
            not_log_df["flops"] / not_log_df["custom_flops"], 3
        )

        # Group data by tensor_network, num_qubits, and cost_fn to compute mean and std dev of log2 operations
        grouped_df_log = (
            df_m.groupby(["tensor_network", "num_qubits", "cost_fn"])["operations_log2"]
            .agg(
                avg_ops="mean",
                std_dev="std",
                min_ops="min",
                max_ops="max",
                num_rows="count",
            )
            .reset_index()
        )

        # Pivot and merge all information for easy plotting
        log_df = grouped_df_log.pivot_table(
            index=["tensor_network", "num_qubits"], columns="cost_fn", values="avg_ops"
        ).reset_index()

        std_df = grouped_df_log.pivot_table(
            index=["tensor_network", "num_qubits"], columns="cost_fn", values="std_dev"
        ).reset_index()

        final_df = add_brute_force_costs(log_df)
        final_df["brute_force_log"] = np.log2(
            final_df["brute_force_cost"].astype(float)
        )

        final_df["custom_std"] = std_df.get(
            "custom_flops", pd.Series([0] * len(final_df))
        )
        final_df["default_std"] = std_df.get("flops", pd.Series([0] * len(final_df)))

        merged = final_df.merge(
            not_log_df[["tensor_network", "num_qubits", "improvement_default_custom"]],
            on=["tensor_network", "num_qubits"],
            how="left",  # keep only df1 rows, ignore extras in df2
        )

        rep_order = [
            "Concatenated Repetition",
            "Holographic",
            "Rotated Surface",
            "Rotated Surface Tanner",
            "Rotated Surface MSP",
            "Hamming Tanner",
            "Hamming MSP",
            "BB Tanner",
            "BB MSP",
        ]
        merged["tensor_network"] = pd.Categorical(
            merged["tensor_network"], categories=rep_order, ordered=True
        )

        merged = merged.sort_values(
            ["tensor_network", "num_qubits"], ascending=[True, True]
        )
        merged.reset_index(inplace=True)

        width = 0.35  # width of the bars
        groups = {}

        for i, row in merged.iterrows():
            bar_height = max(
                row["flops"] + row["default_std"],
                row["custom_flops"] + row["custom_std"],
            )

            y = 1.04 * bar_height

            if row["tensor_network"] == "BB MSP":
                y += 0.05 * bar_height
            if row["tensor_network"] == "Hamming MSP":
                y += 0.05 * bar_height

            groups.setdefault(row["tensor_network"], []).append(i)

            # If there's an optimal entry, we plot 3 bars instead of 2
            if optimal_df is not None:
                match = (
                    (optimal_df["tensor_network"] == row["tensor_network"])
                    & (optimal_df["num_qubits"] == row["num_qubits"])
                ).any()
            else:
                match = False

            if match:
                width = 0.23
                offsets = [-width, 0, width]  # 3 bars evenly spaced around i
            else:
                offsets = [-width / 2, width / 2]  # just 2 bars

            # Custom cost bar + error
            ax.bar(
                i + offsets[-2],
                row["custom_flops"],
                width,
                yerr=row["custom_std"],
                capsize=5,
                color=GROUP_COLORS[row["tensor_network"]],
                edgecolor="black",
                label=f"SST Cost" if i == 0 else "",
            )
            # Default cost bar + error
            ax.bar(
                i + offsets[-1],
                row["flops"],
                width,
                yerr=row["default_std"],
                capsize=5,
                color=GROUP_COLORS[row["tensor_network"]],
                edgecolor="black",
                alpha=0.4,
                label=f"Dense Cost" if i == 0 else "",
            )

            # Brute force cost bar
            ax.bar(
                i,
                float(row["brute_force_log"]),
                width=len(offsets) * width,
                fill=False,  # no fill
                edgecolor="red",
                linestyle="--",
                linewidth=1,
                label="Brute Force" if i == 0 else "",
            )

            if match:

                optimal_row = optimal_df[
                    (optimal_df["tensor_network"] == row["tensor_network"])
                    & (optimal_df["num_qubits"] == row["num_qubits"])
                ].iloc[0]

                y_opt = np.log2(optimal_row["operations_with_bruteforce"])

                # Plot the third (optimal) bar
                ax.bar(
                    i + offsets[0],
                    y_opt,
                    width,
                    color="none",
                    edgecolor="blue",
                    hatch="////",
                    alpha=0.8,
                    label="Optimal Cost" if i == 0 else "",
                )

            # Improvement factor text above bars
            if row["tensor_network"] == "BB MSP":
                i = i + 0.3
            if row["improvement_default_custom"] > 1e3:
                factor_label = f"{row["improvement_default_custom"]:.3e}"  # scientific
            else:
                factor_label = f"{row["improvement_default_custom"]:.3f}"
            ax.text(
                i,
                y,
                f"{factor_label}x",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_xticks(np.arange(len(merged)))
        ax.set_xticklabels(
            [
                f"n={nq}"
                for rep, nq in zip(merged["tensor_network"], merged["num_qubits"])
            ],
            fontsize=11,
        )

        # Transform y-labels to linear number
        yticks = ax.get_yticks()
        ytick_labels = []
        for y in yticks:
            actual_val = 2**y
            exponent = int(np.round(np.log10(actual_val)))
            ytick_labels.append(f"$10^{{{exponent}}}$")

        ax.set_yticklabels(ytick_labels)

        # Make legend colors dark gray and light gray
        legend = ax.legend(fontsize=16, loc="upper center")
        legend.legend_handles[0].set_color("#5A5C5E")
        legend.legend_handles[1].set_color("#94989C")

        # Add group labels below x-ticks
        ymin, _ = ax.get_ylim()
        for group_name, indices in groups.items():
            # Position the text at the center of the group
            x_center = np.mean(indices)
            short_label = RENAME_MAP.get(group_name, group_name)
            ax.text(
                x_center,
                ymin - 16,
                short_label,
                ha="center",
                va="top",
                fontsize=13,
                rotation=60,  # slanted
                rotation_mode="anchor",
            )

        ax.set_ylabel("Contraction Cost", fontsize=18)

        if n_methods > 1:
            method = method.replace("'", "").replace("[", "").replace("]", "")
            ax.set_title(f"{method}", fontsize=20)

    plt.subplots_adjust(bottom=0.1)
    plt.tight_layout()

    output_pdf = out_file.replace(".png", ".pdf")
    plt.savefig(output_pdf, format="pdf", dpi=500)
    plt.close()


def plot_operations_comparison_scatter(data_file, out_file="scatter_comparison.png"):
    """Plots the actual contraction cost vs the estimated cost from cotengra and custom cost function.

    Args:
        data_file: File name that contains data in CSV format.
        out_file: Output file name for the plot.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    axs = axs.flatten()

    df = pd.read_csv(data_file, sep=";")
    representations = df["tensor_network"].unique()

    def plot_by_rep(df, x_label, ax, x_col):
        for rep in representations:
            subdf = df[df["tensor_network"] == rep]
            ax.scatter(
                subdf[x_col],
                subdf["real_operations"],
                label=RENAME_MAP.get(rep, rep),
                color=GROUP_COLORS.get(rep, rep),
            )

        ax.set_xlabel(x_label, fontsize=24)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=18)

    plot_by_rep(df, "Dense Cost", axs[0], "cotengra_cost")
    plot_by_rep(df, "Sparse Stabilizer Tensor Cost", axs[1], "custom_cost")
    axs[0].set_ylabel("Contraction Cost", fontsize=24)

    handles, labels = axs[0].get_legend_handles_labels()

    # Add one legend for the whole figure
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.98, 0.1),
        loc="lower right",
        ncol=1,
        fontsize=20,
        markerscale=2,
    )

    plt.tight_layout()
    output_pdf = out_file.replace(".png", ".pdf")
    plt.savefig(output_pdf, format="pdf", dpi=500)
    plt.close()


def plot_tensor_sparsity_distribution(
    data_file_name, out_file="tensor_sparsity_dist.png"
):
    """Plots the distribution of sparsity of the intermediate tensors.

    Args:
        data_file_name: File name that contains data in CSV format.
        out_file: Output file name for the plot.
    """
    df = pd.read_csv(data_file_name, sep=";")

    # Only plot the largest code from each family
    df = df[df.groupby("network")["num_qubits"].transform("max") == df["num_qubits"]]

    ncols = 3
    representations = df["network"].unique()
    num_reps = len(representations)
    
    nrows = int(num_reps / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows), sharey=True)
    axes = axes.flatten()
    if len(representations) == 1:
        axes = [axes]

    # Set a cutoff value for sparsity to avoid log2(0)
    eps = 1e-6
    df["log2_sparsity"] = np.round(
        np.log2(np.maximum(df["avg_tensor_sparsity"], eps))
    ).astype(int)

    global_min = int(df["log2_sparsity"].min())
    global_max = int(df["log2_sparsity"].max())
    full_range = range(global_min, global_max + 1)

    for i, (ax, rep) in enumerate(zip(axes, REP_ORDER)):
        subset = df[df["network"] == rep]
        nq = int(subset["num_qubits"].iloc[0])

        mean_val = subset["log2_sparsity"].mean()

        counts = subset["log2_sparsity"].value_counts(normalize=True).sort_index()
        counts = counts.reindex(full_range, fill_value=0).sort_index()
        counts.plot(
            kind="bar",
            ax=ax,
            color=GROUP_COLORS.get(rep, rep),
            edgecolor="black",
            alpha=0.9,
        )

        tick_positions = np.arange(len(counts))
        tick_labels = counts.index

        # Show every 2nd tick, starting from the right (index 0)
        ax.set_xticks(tick_positions[::-1][::2][::-1])
        ax.set_xticklabels(tick_labels[::-1][::2][::-1], rotation=0)

        # Find bar that contains the mean
        closest_idx = (np.abs(np.array(counts.index) - mean_val)).argmin()
        ax.axvline(x=closest_idx, color="black", linestyle="--", linewidth=2)

        # Display mean value in a readable format
        val_actual = 1 / (2 ** (-mean_val))
        if val_actual < 1e-2 or val_actual > 1e2:
            label_str = f"mean = {val_actual:.2e}"  # scientific
        else:
            label_str = f"mean = {val_actual:.2f}"

        if rep == "BB Tanner" or rep == "Hamming Tanner":
            x = closest_idx + 5
        else:
            x = closest_idx - 5
        ax.text(
            x,
            0.68,
            f"{label_str}",
            color="black",
            ha="center",
            va="center",
            fontsize=18,
        )

        ax.set_title(f"{RENAME_MAP.get(rep, rep)}, n={nq}", fontsize=20)
        if i >= len(axes) - ncols:  # only display x-labels on bottom row
            ax.set_xlabel(r"$\log_{2}(\text{Tensor Density})$", fontsize=20)
        else:
            ax.set_xlabel("")
        ax.set_ylabel("Probability", fontsize=20)
        ax.tick_params(axis="both", which="major", labelsize=16)

    plt.tight_layout()
    output_pdf = out_file.replace(".png", ".pdf")
    plt.savefig(output_pdf, format="pdf", dpi=500)
    plt.close()


def plot_time_distributions_from_df(
    data,
    bins=30,
    out_file="time_distributions.png",
    method=None,
):
    """
    Plots histogram comparing time distributions for custom_flops vs flops for given data.

    Args:
        data: File name that contains data in CSV format.
        bins: Number of histogram bins. Default is 30.
        out_file: Output file name for the plot.
        method: If specified, filters data to only include this method.
    """
    df = pd.read_csv(data, sep=";")
    plt.figure(figsize=(10, 6))

    df = df[
        ~(
            (df["tensor_network"] == "BB MSP")
            & (df["num_qubits"] == 30)
            & (df["methods"] == "['kahypar']")
        )
    ]  # remove BB MSP with 30 qubits, don't have full data

    if method is not None:
        df = df[df["methods"].apply(lambda x: f"{method}" in x)]

    # Find the max num_qubits per tensor_network
    max_qubits_df = df.loc[df.groupby("tensor_network")["num_qubits"].idxmax()]
    networks = max_qubits_df["tensor_network"].unique()
    n_networks = len(networks)

    n_cols = min(3, n_networks)  # up to 3 columns
    assert n_cols > 0, "No networks to plot."

    n_rows = math.ceil(n_networks / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # Iterate over each tensor_network
    for i, network in enumerate(REP_ORDER):
        ax = axes[i]
        sub_df = df[
            (df["tensor_network"] == network)
            & (
                df["num_qubits"]
                == df[df["tensor_network"] == network]["num_qubits"].max()
            )
        ]

        custom_flops_times = sub_df[sub_df["cost_fn"] == "custom_flops"]["time"]
        flops_times = sub_df[sub_df["cost_fn"] == "flops"]["time"]
        nq = int(sub_df["num_qubits"].iloc[0])

        if not custom_flops_times.empty:
            counts, bin_edges = np.histogram(custom_flops_times, bins=bins)
            ax.bar(
                bin_edges[:-1],
                counts / counts.sum(),
                width=np.diff(bin_edges),
                alpha=1,
                color=GROUP_COLORS[network],
                label="SST",
            )

        if not flops_times.empty:
            counts, bin_edges = np.histogram(flops_times, bins=bins)
            ax.bar(
                bin_edges[:-1],
                counts / counts.sum(),
                width=np.diff(bin_edges),
                alpha=0.6,
                color="gray",
                label="Dense",
            )

        ax.set_xlabel("Time", fontsize=20)
        ax.set_ylabel("Probability", fontsize=20)
        ax.set_title(f"{RENAME_MAP.get(network, network)}, n={nq}", fontsize=22)
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.legend(fontsize=16)

    plt.tight_layout()
    output_pdf = out_file.replace(".png", ".pdf")
    plt.savefig(output_pdf, format="pdf", dpi=500)
    plt.close()


def plot_log_tensor_size_vs_open_legs(data, out_file="tensor_size_vs_open_legs.png"):
    """
    Plot log₄(tensor_size) vs number of open legs for each family.
    Separate plots for actual_tensor_size and dense_tensor_size.
    """
    # Set default figure size
    df = pd.read_csv(data, sep=";")

    numeric_columns = [
        "num_qubits",
        "num_run",
        "num_open_legs",
        "actual_tensor_size",
        "dense_tensor_size",
        "avg_tensor_sparsity",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    plt.rcParams["figure.figsize"] = [6, 4]

    # Create family identifier (only by network/code family, not qubit count)
    df["family"] = df["network"]

    # Calculate log₄ of tensor sizes
    df["log4_actual_size"] = np.log(df["actual_tensor_size"]) / np.log(4)
    df["log4_dense_size"] = np.log(df["dense_tensor_size"]) / np.log(4)

    families = df["family"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(families)))

    # Combined plot: log₄(actual_tensor_size) vs open legs
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot the 45-degree line representing dense tensor size
    max_open_legs = df["num_open_legs"].max()
    ax.plot(
        [0, max_open_legs],
        [0, max_open_legs],
        color="black",
        linestyle="--",
        linewidth=2,
        label="dense tensor size",
        alpha=0.8,
    )

    for idx, family in enumerate(sorted(families)):
        family_data = df[df["family"] == family]

        # Plot only actual tensor size with circles
        ax.scatter(
            family_data["num_open_legs"],
            family_data["log4_actual_size"],
            label=family,
            color=colors[idx],
            marker="o",
            alpha=0.7,
            s=40,
        )

    ax.set_xlabel("Number of Open Legs", fontsize=16)
    ax.set_ylabel("log₄(Tensor Size)", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=14)

    plt.tight_layout()

    output_pdf = out_file.replace(".png", ".pdf")
    plt.savefig(output_pdf, format="pdf", dpi=500)
    plt.close()


def main():
    plot_time_distributions_from_df(
        "results/data/64_trials_results.csv",
        out_file="results/images/time_comparison_kahypar.png",
        method="kahypar",
    )

    plot_log_operations_bar_chart(
        "results/data/64_trials_results.csv",
        out_file="results/images/bar_chart_log_64_trials_greedy.png",
        method="greedy",
    )

    plot_tensor_sparsity_distribution(
        "results/data/tensor_sparsity_info.csv",
        out_file="results/images/tensor_sparsity_dist.png",
    )

    plot_operations_comparison_scatter(
        "results/data/wep_calculations_operations_comparison.csv",
        out_file="results/images/scatter_plot_comparison.png",
    )

    plot_log_tensor_size_vs_open_legs(
        "results/data/tensor_sparsity_info.csv",
        out_file="results/images/tensor_size_vs_open_legs.png",
    )

if __name__ == "__main__":
    main()
