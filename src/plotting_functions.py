import ast
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns

group_colors = {
    "Concatenated Repetition": "#2B8A20",  
    "Rotated Surface": "#F4320B",  
    "Rotated Surface MSP": "#DBB71D", 
    "Rotated Surface Tanner": "#F238D5", 
    "Hamming MSP": "#27BAA3",
    "Hamming Tanner": "#278BF5",
    "Holographic": "#695454",
    "BB MSP": "#6E4BDE",
    "BB Tanner": "#4B72DE"
}

rename_map = {
    "Rotated Surface MSP": "RSC MSP",
    "Rotated Surface Tanner": "RSC Tanner",
    "Rotated Surface": "RSC",
    "Hamming MSP": "Ham MSP",
    "Hamming Tanner": "Ham Tanner",
    "Concatenated Repetition": "Concat Rep",
}

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
    df["brute_force_cost"] = df.set_index(["tensor_network", "num_qubits"]).index.map(mapping)
    return df


def plot_improvement_factor_bar_chart(
    contraction_cost_file, out_file="bar_chart_comparison.png"
):
    df = pd.read_csv(contraction_cost_file, sep=";")
    df = df[df["methods"] != "['labels']"]

    methods = sorted(df["methods"].unique())
    print("methods are:", methods)
    n_methods = len(methods)
    ncols = 1
    nrows = math.ceil(n_methods / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 6 * nrows))
    axes = axes.flatten()

    for idx, method in enumerate(methods):
        ax = axes[idx]
        df_m = df[df["methods"] == method]
        df_m = df_m[~((df_m["tensor_network"] == "Hamming MSP") & (df_m["num_qubits"] == 31))]

        grouped_df = (
            df_m.groupby(["tensor_network", "num_qubits", "cost_fn"])["operations"]
            .agg(avg_ops="mean", std_ops="std", num_rows="count")
            .reset_index()
        )

        final_df = grouped_df.pivot_table(
            index=["tensor_network", "num_qubits"], columns="cost_fn", values="avg_ops"
        ).reset_index()
        std_df = grouped_df.pivot_table(
            index=["tensor_network", "num_qubits"], columns="cost_fn", values="std_ops"
        ).reset_index()

        final_df = add_brute_force_costs(final_df)
        
        final_df["custom_std"] = std_df.get("custom_flops", pd.Series([0]*len(final_df)))
        final_df["default_std"] = std_df.get("flops", pd.Series([0]*len(final_df)))

        # Calculated improvement factors
        final_df["improvement_bruteforce_custom"] = (
            final_df["brute_force_cost"].astype(float) / final_df["custom_flops"]
        )
        final_df["improvement_bruteforce_default"] = (
            final_df["brute_force_cost"].astype(float) / final_df["flops"]
        )
        final_df["improvement_default_custom"] = final_df["flops"] / final_df["custom_flops"]

        # approximate relative std propagation
        final_df["err_custom"] = final_df["custom_std"] / final_df["custom_flops"] * final_df["improvement_bruteforce_custom"]
        final_df["err_default"] = final_df["default_std"] / final_df["flops"] * final_df["improvement_bruteforce_default"]


        print(final_df)

        # Sort by representation based on smallest improvement factor
        rep_order = (
            final_df.groupby("tensor_network")["improvement_bruteforce_custom"]
            .min()
            .sort_values()
            .index
        )
        final_df["tensor_network"] = pd.Categorical(
            final_df["tensor_network"], categories=rep_order, ordered=True
        )
        final_df = final_df.sort_values(["tensor_network", "num_qubits"], ascending=[True, True])
        final_df.reset_index(inplace=True)

        # Define colors based on representation
        reps = final_df["tensor_network"].unique()
        cmap = cm.get_cmap("tab10", len(reps))
        rep_colors = {rep: cmap(i) for i, rep in enumerate(reps)}

        #fig, ax = plt.subplots(figsize=(14, 8))
        width = 0.35  # width of the bars
        groups = {}

        for i, row in final_df.iterrows():
            # Determine the height for placing the text above the bars
            bar_height = max(
                row["improvement_bruteforce_custom"], row["improvement_bruteforce_default"]
            )
            bar_err = max(
                row["improvement_bruteforce_custom"], row["improvement_bruteforce_default"]
            )
            y = bar_height + bar_err + 0.05 * bar_height

            base_color = rep_colors[row["tensor_network"]]
            groups.setdefault(row["tensor_network"], []).append(i)


            # Custom cost bar + error
            ax.bar(
                i - width / 2,
                row["improvement_bruteforce_custom"],
                width,
                yerr=row["err_custom"],
                capsize=5,
                color=mcolors.to_rgba(base_color),
                edgecolor="black",
                label=f"Custom Stabilizer Cost" if i == 0 else "",
            )
            # Default cost bar + error
            ax.bar(
                i + width / 2,
                row["improvement_bruteforce_default"],
                width,
                yerr=row["err_default"],
                capsize=5,
                color=mcolors.to_rgba(base_color),
                edgecolor="black",
                alpha=0.4,
                label=f"Default Flops Cost" if i == 0 else "",
            )

            ax.text(
                i,
                y,
                f"{row['improvement_default_custom']:.2f}x",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.axhline(1, color="red", linestyle="--", label="No improvement")
        ax.set_xticks(np.arange(len(final_df)))
        ax.set_xticklabels(
            [f"n={nq}" for rep, nq in zip(final_df["tensor_network"], final_df["num_qubits"])]
        )
        ax.set_ylabel("Default Operations / Cotengra")
        ax.set_yscale("log")
        ax.set_title("Improvement Factor for Cotengra vs Brute Force WEP Calculation: " + method + " high level")
        ax.legend()

        rename_map = {
            "Rotated Surface MSP": "RSC MSP",
            "Rotated Surface Tanner": "RSC Tanner",
            "Rotated Surface": "RSC",
            "Hamming MSP": "Ham MSP",
            "Hamming Tanner": "Ham Tanner",
            "Concatenated Repetition": "Concat Rep",
        }

        # Add group labels below x-ticks
        ymin, _ = ax.get_ylim()
        for group_name, indices in groups.items():
            # Position the text at the center of the group
            x_center = np.mean(indices)
            short_label = rename_map.get(group_name, group_name)
            ax.text(x_center, ymin * 0.005, short_label, ha="center", va="top", fontsize=10)

    plt.subplots_adjust(bottom=0.1)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_absolute_operations_bar_chart(
    contraction_cost_file, out_file="bar_chart_operations_comparison.png"
):
    df = pd.read_csv(contraction_cost_file, sep=";")
    df = df[df["methods"] != "['kahypar']"]

    methods = sorted(df["methods"].unique())
    n_methods = len(methods)
    nrows = 1
    ncols = math.ceil(n_methods / nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12 * ncols, 6), sharey=True)
    #axes = axes.flatten()

    for idx, method in enumerate(methods):
        #ax = axes[idx]
        ax = axes
        df_m = df[df["methods"] == method]

        grouped_df = (
            df_m.groupby(["tensor_network", "num_qubits", "cost_fn"])["operations"]
            .agg(
                avg_ops="mean", 
                std_ops="std", 
                min_ops="min",
                max_ops="max",
                num_rows="count")
            .reset_index()
        )

        final_df = grouped_df.pivot_table(
            index=["tensor_network", "num_qubits"], columns="cost_fn", values="avg_ops"
        ).reset_index()
        std_df = grouped_df.pivot_table(
            index=["tensor_network", "num_qubits"], columns="cost_fn", values="std_ops"
        ).reset_index()
        min_df = grouped_df.pivot_table(
            index=["tensor_network", "num_qubits"], columns="cost_fn", values="min_ops"
        ).reset_index()

        max_df = grouped_df.pivot_table(
            index=["tensor_network", "num_qubits"], columns="cost_fn", values="max_ops"
        ).reset_index()

        final_df = add_brute_force_costs(final_df)
        
        final_df["custom_std"] = std_df.get("custom_flops", pd.Series([0]*len(final_df)))
        final_df["default_std"] = std_df.get("flops", pd.Series([0]*len(final_df)))
        final_df["custom_min"] = min_df["custom_flops"]
        final_df["default_min"] = min_df["flops"]
        final_df["custom_max"] = max_df["custom_flops"]
        final_df["default_max"] = max_df["flops"]

        # Sort by representation based on smallest operations
        rep_order = ["Concatenated Repetition", "Holographic", "Rotated Surface", "Rotated Surface Tanner", "Rotated Surface MSP", "Hamming Tanner", "Hamming MSP", "BB Tanner", "BB MSP"]
        final_df["tensor_network"] = pd.Categorical(
            final_df["tensor_network"], categories=rep_order, ordered=True
        )
        final_df["improvement_default_custom"] = final_df["flops"] / final_df["custom_flops"]
        final_df = final_df.sort_values(["tensor_network", "num_qubits"], ascending=[True, True])
        final_df.reset_index(inplace=True)

        #fig, ax = plt.subplots(figsize=(14, 8))
        width = 0.35  # width of the bars
        groups = {}

        for i, row in final_df.iterrows():
            bar_height = max(
                row["flops"], row["custom_flops"]
            )
            max_height = max(
                row["custom_max"], row["default_max"]
            )
            y = 2*bar_height + max_height

            if(row["tensor_network"] == "BB MSP"):
                y += 3*max_height
            if(row["tensor_network"] == "Hamming MSP"):
                y += 6*max_height
            
            
            groups.setdefault(row["tensor_network"], []).append(i)

            yerr_custom = np.array([[row["custom_flops"]- row["custom_min"]], [row["custom_max"] - row["custom_flops"]]])
            yerr_default = np.array([[row["flops"]- row["default_min"]], [row["default_max"] - row["flops"]]])

            # Custom cost bar + error
            ax.bar(
                i - width / 2,
                row["custom_flops"],
                width,
                yerr=yerr_custom,
                capsize=5,
                color=group_colors[row["tensor_network"]],
                edgecolor="black",
                label=f"SST Cost" if i == 0 else "",
            )
            # Default cost bar + error
            ax.bar(
                i + width / 2,
                row["flops"],
                width,
                yerr=yerr_default,
                capsize=5,
                color=group_colors[row["tensor_network"]],
                edgecolor="black",
                alpha=0.4,
                label=f"Dense Cost" if i == 0 else "",
            )

            ax.bar(
                i,  # center of the brute force bar
                float(row["brute_force_cost"]),
                width=2*width,  # skinny so it doesn’t overwhelm
                fill=False,  # no fill
                edgecolor="red",
                linestyle="--",
                linewidth=1,
                label="Brute Force" if i == 0 else ""
            )   

            ax.text(
                i,
                y,
                f"{row['improvement_default_custom']:.2f}x",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )


        ax.set_xticks(np.arange(len(final_df)))
        ax.set_xticklabels(
            [f"n={nq}" for rep, nq in zip(final_df["tensor_network"], final_df["num_qubits"])], fontsize=11
        )
        
        ax.set_yscale("log")
        ax.legend(fontsize=16, loc='upper center')

        # Add group labels below x-ticks
    ymin, _ = ax.get_ylim()
    for group_name, indices in groups.items():
        # Position the text at the center of the group
        x_center = np.mean(indices)
        short_label = rename_map.get(group_name, group_name)
        axes.text(x_center, ymin * 0.00005, short_label, ha="center", va="top", fontsize=13, rotation=60,      # slanted
    rotation_mode="anchor")


    axes.set_ylabel("Contraction Cost", fontsize=18)
    plt.subplots_adjust(bottom=0.1)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    
    


def plot_operations_comparison_scatter(
    default_data_file, custom_data_file, out_file="scatter_comparison.png"
):
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    axs = axs.flatten()

    df = pd.read_csv(default_data_file, sep=";")
    df_custom = pd.read_csv(custom_data_file, sep=";")

    # Map each representation to a color
    representations = df["representation"].unique()
    colors = plt.cm.tab10.colors
    color_map = {rep: colors[i % len(colors)] for i, rep in enumerate(representations)}

    def plot_by_rep(df, x_label, ax):
        for rep in representations:
            subdf = df[df["representation"] == rep]
            ax.scatter(
                2 ** subdf["score_cotengra"],
                subdf["operations"],
                label=rep,
                color=color_map[rep],
            )

        ax.set_xlabel(x_label, fontsize=14)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)

    plot_by_rep(df, "Flops Cotengra Cost", axs[0])
    plot_by_rep(df_custom, "Custom Stabilizer Cost", axs[1])
    axs[0].set_ylabel("Operations", fontsize=14)

    handles, labels = axs[0].get_legend_handles_labels()

    # Add one legend for the whole figure
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.09, 0.9),
        loc="upper left",
        ncol=1,
        fontsize=12,
    )

    fig.suptitle("Contraction Cost (Operations) vs Cotengra Costs", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_operations_comparison_scatter_same_file(
    data_file, out_file="scatter_comparison.png"
):
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
                label=rename_map.get(rep, rep),
                color=group_colors.get(rep, rep),
            )

        ax.set_xlabel(x_label, fontsize=24)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=18)

    plot_by_rep(df, "Cotengra Flops Cost", axs[0], "cotengra_cost")
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
        markerscale=2
    )
    

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()



def plot_operations_scatter_same_plot(
    default_data_file, custom_data_file, out_file="scatter_comparison.png"
):
    fig, ax = plt.subplots(figsize=(12, 8))

    df_default = pd.read_csv(default_data_file, sep=";")
    df_custom = pd.read_csv(custom_data_file, sep=";")

    # keep all rows with the minimal num_qubits for each representation
    df_default = df_default[
        df_default.groupby("representation")["num_qubits"].transform("min")
        == df_default["num_qubits"]
    ]
    df_custom = df_custom[
        df_custom.groupby("representation")["num_qubits"].transform("min")
        == df_custom["num_qubits"]
    ]

    # Unique representations
    representations = sorted(
        set(df_default["representation"]).union(df_custom["representation"])
    )

    # Assign each rep a color
    colors = plt.cm.tab10.colors
    color_map = {rep: colors[i % len(colors)] for i, rep in enumerate(representations)}

    # Plot default with circles
    for rep in representations:
        subdf = df_default[df_default["representation"] == rep]
        nq = int(subdf["num_qubits"].iloc[0])
        ax.scatter(
            2 ** subdf["score_cotengra"],
            subdf["operations"],
            label=f"{rep} (default), n={nq}",
            color=color_map[rep],
            alpha=0.4,
            marker="x",  # circle
        )

    # Plot custom with squares
    for rep in representations:
        subdf = df_custom[df_custom["representation"] == rep]
        nq = int(subdf["num_qubits"].iloc[0])
        ax.scatter(
            2 ** subdf["score_cotengra"],
            subdf["operations"],
            label=f"{rep} (custom), n={nq}",
            color=color_map[rep],
            marker="s",  # square
        )

    # Axes formatting
    ax.set_xlabel("Cotengra Cost", fontsize=14)
    ax.set_ylabel("Operations", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)

    # Legend
    ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")

    # Title + layout
    fig.suptitle("Contraction Cost (Operations) vs Cotengra Cost", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


def plot_tensor_sparsity_distribution(data_file_name, out_file="tensor_sparsity_dist.png"):
    df = pd.read_csv(data_file_name, sep=";")

    # Count frequencies of sparsity values
    df["avg_tensor_sparsity"] = df["avg_tensor_sparsity"].round(4)

    # Only plot the largest code from each family
    df = df[
        df.groupby("network")["num_qubits"].transform("max")
        == df["num_qubits"]
    ]

    ncols = 3
    representations = df["network"].unique()
    num_reps = len(representations)
    nrows = int(num_reps/ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5*nrows), sharey=True)
    axes = axes.flatten()
    if len(representations) == 1:
        axes = [axes]

    # bins = np.linspace(df["avg_tensor_sparsity"].min(),
    #                 df["avg_tensor_sparsity"].max(),
    #                 20 + 1)

    for ax, rep in zip(axes, representations):
        subset = df[df["network"] == rep]

        nq = int(subset["num_qubits"].iloc[0])
        # use ax.hist instead of value_counts
        # ax.hist(subset["avg_tensor_sparsity"],
        #         bins=bins,
        #         density=True,              # probability instead of raw counts
        #         color=group_colors.get(rep, rep),
        #         alpha=0.7,                 # optional transparency
        #         edgecolor="black") 

        counts = subset["avg_tensor_sparsity"].value_counts(normalize=True).sort_index()
        counts.plot(kind="bar", ax=ax, color=group_colors.get(rep, rep), edgecolor="black", alpha=0.9)

        ax.set_title(f"{rename_map.get(rep, rep)}, n={nq}", fontsize=20)
        ax.set_xlabel("Tensor Density", fontsize=20)
        ax.set_ylabel("Probability", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_time_distributions_from_df(data, bins=30, density=True):
    """
    Plots histogram + KDE curves comparing time distributions
    for custom_flops vs flops from a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: 'cost_fn' and 'time'.
    bins : int, optional
        Number of histogram bins. Default 30.
    density : bool, optional
        Normalize histograms to probability densities. Default True.
    """
    df = pd.read_csv(data, sep=";")
    plt.figure(figsize=(10,6))

    df = df[df["methods"].apply(lambda x: 'greedy' in x)]
    print(df)
    
    # Find the max num_qubits per tensor_network
    max_qubits_df = df.loc[df.groupby("tensor_network")["num_qubits"].idxmax()]
    networks = max_qubits_df["tensor_network"].unique()
    n_networks = len(networks)

    n_cols = min(3, n_networks)  # up to 3 columns
    n_rows = math.ceil(n_networks / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    axes = axes.flatten() 

    # Iterate over each tensor_network
    for i, network in enumerate(networks):
        ax = axes[i]
        sub_df = df[(df["tensor_network"] == network) & 
                    (df["num_qubits"] == df[df["tensor_network"] == network]["num_qubits"].max())]

        custom_flops_times = sub_df[sub_df["cost_fn"] == "custom_flops"]["time"]
        flops_times = sub_df[sub_df["cost_fn"] == "flops"]["time"]

        if not custom_flops_times.empty:
            ax.hist(custom_flops_times, bins=bins, density=density,
                    alpha=0.4, color="blue")
            sns.kdeplot(custom_flops_times, ax=ax, label="custom flops", color="blue", lw=2)
        if not flops_times.empty:
            ax.hist(flops_times, bins=bins, density=density,
                    alpha=0.4, color="orange")
            sns.kdeplot(flops_times, ax=ax, label="default flops", color="orange", lw=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Density" if density else "Count")
        ax.set_title(f"{network} (largest num_qubits)")
        ax.legend()

    fig.suptitle("Cotengra Searching Times Default vs Custom (Greedy)", fontsize=16)
    plt.tight_layout()
    plt.savefig("cotengra_time_comparison_greedy.png", bbox_inches="tight")
    plt.close()


plot_time_distributions_from_df("outputs/results_9_25/merged_results_5_min_cutoff_9_25.csv")

# plot_absolute_operations_bar_chart(
#     "results/64_trials_results_fixed_10_1.csv",
#     "bar_chart_absolute_64_trials_10_1_greedy.png",
# )

# plot_tensor_sparsity_distribution("results/tensor_sparsity_info.csv", "tensor_sparsity_dist_counts.png")


# plot_operations_comparison_scatter_same_file("results/wep_calculations_operations_comparison.csv")