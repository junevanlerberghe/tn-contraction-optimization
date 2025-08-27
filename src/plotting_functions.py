import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def add_brute_force_costs(df):
    mapping = {
        ("Concatenated", 9): 2**(9 - 1),
        ("Concatenated", 25): 2**(25 - 1),
        ("Concatenated", 49): 2**(49 - 1),
        ("Rotated Surface", 9): 2**(9 - 1),
        ("Rotated Surface", 25): 2**(25 - 1),
        ("Rotated Surface", 49): 2**(49 - 1),
        ("Rotated Surface", 81): 2**(81 - 1),
        ("Rotated Surface MSP", 9): 2**(9 - 1),
        ("Rectangular Surface", 9): 2**(9 - 1),
        ("Rectangular Surface", 15): 2**(15 - 1),
        ("Rectangular Surface", 21): 2**(21 - 1),
        ("Hamming MSP", 7): 2**(7 - 1),
        ("Hamming MSP", 15): 2**(15 - 7),
        ("Hamming MSP", 31): 2**(31 - 21),
        ("Holographic", 25) : 2**(25 - 11),
        ("Holographic", 95) : 2**(95 - 51),
        ("BB MSP", 18): 2**(18 - 4),
        ("BB MSP", 30): 2**(30 - 4),
    }
    df["brute_force_cost"] = df.set_index(["network", "num_qubits"]).index.map(mapping)
    return df


def plot_improvement_factor_bar_chart(contraction_cost_file, out_file="bar_chart_comparison.png"):
    df = pd.read_csv(contraction_cost_file, sep=";")

    grouped_df = (
        df.groupby(["network", "num_qubits", "cost_fn"])["operations"]
        .agg(avg_ops='mean', std_ops='std', num_rows='count')
        .reset_index()
    )
    
    final_df = grouped_df.pivot_table(
        index=["network", "num_qubits"],
        columns="cost_fn",
        values="avg_ops"
    ).reset_index()

    final_df = add_brute_force_costs(final_df)
    print(final_df)

    # Calculated improvement factors 
    final_df["improvement_bruteforce_custom"] = final_df["brute_force_cost"].astype(float) / final_df["custom"]
    final_df["improvement_bruteforce_default"] = final_df["brute_force_cost"].astype(float) / final_df["combo"]
    final_df["improvement_default_custom"] = final_df["combo"] / final_df["custom"]

    print(final_df)

    # Sort by representation based on smallest improvement factor
    rep_order = (
        final_df.groupby("network")["improvement_bruteforce_custom"]
        .min()   
        .sort_values()  
        .index
    )
    final_df["network"] = pd.Categorical(final_df["network"], categories=rep_order, ordered=True)
    final_df = final_df.sort_values(["network", "num_qubits"], ascending=[True, True])
    final_df.reset_index(inplace=True)

    # Define colors based on representation
    reps = final_df["network"].unique()
    cmap = cm.get_cmap("tab10", len(reps))
    rep_colors = {rep: cmap(i) for i, rep in enumerate(reps)}

    fig, ax = plt.subplots(figsize=(14, 8))
    width = 0.35  # width of the bars
    groups = {}

    for i, row in final_df.iterrows():
        # Determine the height for placing the text above the bars
        bar_height = max(row["improvement_bruteforce_custom"], row["improvement_bruteforce_default"])
        bar_err = max(row["improvement_bruteforce_custom"], row["improvement_bruteforce_default"])
        y = bar_height + bar_err + 0.05 * bar_height

        base_color = rep_colors[row["network"]]
        groups.setdefault(row["network"], []).append(i)

        ax.bar(i - width/2, row["improvement_bruteforce_custom"], width, color=mcolors.to_rgba(base_color), edgecolor="black", label=f"Custom Stabilizer Cost" if i == 0 else "")
        ax.bar(i + width/2, row["improvement_bruteforce_default"], width, color=mcolors.to_rgba(base_color), edgecolor="black", alpha=0.4, label=f"Default Combo Cost" if i == 0 else "")

        ax.text(
            i, y,
            f"{row['improvement_default_custom']:.2f}x",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.axhline(1, color="red", linestyle="--", label="No improvement")
    ax.set_xticks(np.arange(len(final_df)))
    ax.set_xticklabels(
        [f"n={nq}" for rep, nq in zip(final_df['network'], final_df['num_qubits'])]
    )
    ax.set_ylabel("Default Operations / Cotengra")
    ax.set_yscale("log")
    ax.set_title("Improvement Factor for Cotengra vs Brute Force WEP Calculation")
    ax.legend()

    # Add group labels below x-ticks
    ymin, _ = ax.get_ylim()
    for group_name, indices in groups.items():
        # Position the text at the center of the group
        x_center = np.mean(indices)
        ax.text(x_center, ymin * 0.05, group_name, ha='center', va='top', fontsize=10)

    plt.subplots_adjust(bottom=0.1)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()



def plot_operations_comparison_scatter(default_data_file, custom_data_file, out_file="scatter_comparison.png"):
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
            ax.scatter(2**subdf["score_cotengra"], subdf["operations"], label=rep, color=color_map[rep])
        
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)

    plot_by_rep(df, "Combo Cotengra Cost", axs[0])
    plot_by_rep(df_custom, "Custom Stabilizer Cost", axs[1])
    axs[0].set_ylabel("Operations", fontsize=14)

    handles, labels = axs[0].get_legend_handles_labels()

    # Add one legend for the whole figure
    fig.legend(handles, labels, bbox_to_anchor=(0.09, 0.9), loc='upper left', ncol=1, fontsize=12)

    fig.suptitle("Contraction Cost (Operations) vs Cotengra Costs", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_operations_scatter_same_plot(default_data_file, custom_data_file, out_file="scatter_comparison.png"):
    fig, ax = plt.subplots(figsize=(12, 8))

    df_default = pd.read_csv(default_data_file, sep=";")
    df_custom = pd.read_csv(custom_data_file, sep=";")

    # keep all rows with the minimal num_qubits for each representation
    df_default = df_default[df_default.groupby("representation")["num_qubits"].transform("min") == df_default["num_qubits"]]
    df_custom = df_custom[df_custom.groupby("representation")["num_qubits"].transform("min") == df_custom["num_qubits"]]

    # Unique representations
    representations = sorted(set(df_default["representation"]).union(df_custom["representation"]))

    # Assign each rep a color
    colors = plt.cm.tab10.colors
    color_map = {rep: colors[i % len(colors)] for i, rep in enumerate(representations)}

    # Plot default with circles
    for rep in representations:
        subdf = df_default[df_default["representation"] == rep]
        nq = int(subdf["num_qubits"].iloc[0])
        ax.scatter(
            2**subdf["score_cotengra"],
            subdf["operations"],
            label=f"{rep} (default), n={nq}",
            color=color_map[rep],
            alpha=0.4,
            marker="x"   # circle
        )

    # Plot custom with squares
    for rep in representations:
        subdf = df_custom[df_custom["representation"] == rep]
        nq = int(subdf["num_qubits"].iloc[0]) 
        ax.scatter(
            2**subdf["score_cotengra"],
            subdf["operations"],
            label=f"{rep} (custom), n={nq}",
            color=color_map[rep],
            marker="s"   # square
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



plot_operations_scatter_same_plot("outputs/results_8_14/tn_architectures_calc_default.csv",
                                   "outputs/results_8_14/tn_architectures_calc.csv",
                                "scatter_results_both.png")