import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def merge_data(df_default, df_custom):
    default_stats = (
        df_default.groupby(["representation", "num_qubits", "brute_force"])["operations_w_bruteforce"]
        .agg(default_avg_ops='mean', default_std_ops='std', default_num_rows='count')
        .reset_index()
    )

    custom_stats = (
        df_custom.groupby(["representation", "num_qubits", "brute_force"])["operations_w_bruteforce"]
        .agg(custom_avg_ops='mean', custom_std_ops='std', custom_num_rows='count')
        .reset_index()
    )

    final_df = pd.merge(default_stats, custom_stats, on=["representation", "num_qubits", "brute_force"])
    final_df = final_df.sort_values("default_avg_ops", ascending=False)
    return final_df


def plot_improvement_factor_bar_chart(default_data_file, custom_data_file, out_file="bar_chart_comparison.png"):
    df_default = pd.read_csv(default_data_file, sep=";")
    df_custom = pd.read_csv(custom_data_file, sep=";")
    
    final_df = merge_data(df_default, df_custom)

    # Calculated improvement factors 
    final_df["improvement_bruteforce_custom"] = final_df["brute_force"].astype(float) / final_df["custom_avg_ops"]
    final_df["improvement_bruteforce_default"] = final_df["brute_force"].astype(float) / final_df["default_avg_ops"]
    final_df["improvement_default_custom"] = final_df["default_avg_ops"] / final_df["custom_avg_ops"]

    # Sort by representation based on smallest improvement factor
    rep_order = (
        final_df.groupby("representation")["improvement_bruteforce_custom"]
        .min()   
        .sort_values()  
        .index
    )
    final_df["representation"] = pd.Categorical(final_df["representation"], categories=rep_order, ordered=True)
    final_df = final_df.sort_values(["representation", "num_qubits"], ascending=[True, True])
    final_df.reset_index(inplace=True)

    # Define colors based on representation
    reps = final_df["representation"].unique()
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

        base_color = rep_colors[row["representation"]]
        groups.setdefault(row["representation"], []).append(i)

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
        [f"n={nq}" for rep, nq in zip(final_df['representation'], final_df['num_qubits'])]
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

    plot_by_rep(df, "Combo Cotengra Cost", axs[0], "")
    plot_by_rep(df_custom, "Custom Stabilizer Cost", axs[1], "")
    axs[0].set_ylabel("Operations", fontsize=14)

    handles, labels = axs[0].get_legend_handles_labels()

    # Add one legend for the whole figure
    fig.legend(handles, labels, bbox_to_anchor=(0.07, 0.9), loc='upper left', ncol=1, fontsize=14)

    fig.suptitle("Contraction Cost (Operations) vs Cotengra Costs", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()