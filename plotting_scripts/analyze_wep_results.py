import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def visualize_time_vs_metrics(df,
                              outpath="outputs/images/time_vs_metrics_operations.png"):

    metrics = [
        "score_cotengra",
        "flops",
        "write"
    ]

    metrics2 = [
        "New PTE Calc",
        "total_operations",
        "operations_estimate"
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(metrics),
        figsize=(5 * len(metrics), 5*2)
    )

    for ax, metric in zip(axes[0], metrics):
        ax.scatter(df[metric], df["contraction_time"], alpha=0.7)
        ax.set_xlabel(metric)
        ax.set_ylabel("contraction_time")
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        ax.set_ylim(0, 10)
        ax.set_title(f"Time vs {metric}")
        ax.grid(True)

    for ax, metric in zip(axes[1], metrics2):
        ax.scatter(df[metric], df["contraction_time"], alpha=0.7)
        ax.set_xlabel(metric)
        ax.set_ylabel("contraction_time")
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        ax.set_ylim(0, 10)
        if metric == "total_operations":
            ax.set_xlim(0, 40000)
        ax.set_title(f"Time vs {metric}")
        ax.grid(True)

    for leftover_ax in axes[5:]:
        leftover_ax.axis("off")

    fig.suptitle("Contraction Time vs Various Metrics for Surface Code, all representations, d=3", fontsize=16)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_distance_vs_time_for_qshor(df, filename, q_shor=None):
    ncols = 2
    q_shors = sorted(d for d in df["q_shor"].unique())
    if q_shor is not None:
        q_shors = [q_shor]

    nplots = len(q_shors)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()  # for easy indexing

    for ax, q_shor in zip(axes, q_shors):
        # Filter data for the specified q_shor
        filtered = df[df['q_shor'] == q_shor]
        
        # Group by distance and representation, then average time
        grouped = filtered.groupby(['distance', 'representation'])['avg time'].mean().reset_index()
        
        # Pivot to get representations as columns
        pivot = grouped.pivot(index='distance', columns='representation', values='avg time')

        for rep in pivot.columns:
            ax.plot(pivot.index, pivot[rep], marker='o', label=rep)

        ax.set_xlabel('Distance')
        ax.set_ylabel('Average Time (s) - log scale')
        ax.set_yscale('log')
        ax.set_title(f'Contraction Time vs. Distance \n(p_flip = {q_shor})')
        ax.legend()
        ax.grid(True)

    fig.suptitle("100 runs for d = 3, 30 runs for d = 4. Data collection in progress", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

        

def plot_q_shor_vs_time(df, filename, distance=None):
    if distance is not None:
        distances = [distance]
    else:
        distances = sorted(d for d in df["distance"].unique())
    ncols = 2
    nplots = len(distances)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()  # for easy indexing

    for ax, distance_val in zip(axes, distances):
        group = df[df["distance"] == distance_val]
        grouped = group.groupby(["representation", 'q_shor'])[['avg time', 'std dev']].mean().reset_index()
        representations = grouped["representation"].unique()

        for rep in representations:
            #ax.plot(pivot.index, pivot[rep], marker='o', label=rep)
            rep_data = grouped[grouped["representation"] == rep]
            x = rep_data["q_shor"]
            y = rep_data["avg time"]
            yerr = rep_data["std dev"]
            ax.plot(x, y, marker='o', label=rep)
            yerr_diff = y - yerr
            if any(yerr_diff < 0):
                yerr_diff[yerr_diff < 0] = 0
            #ax.fill_between(x, yerr_diff, y + yerr, alpha=0.2)  # shaded std dev

        ax.set_xlabel('p_flip (0.0 = surface, 1.0 = shor)')
        ax.set_ylabel('Average Time (s)')
        ax.set_yscale('log')
        ax.set_title(f'Distance = {distance_val}')
        ax.legend(fontsize='x-small', markerscale=0.5, handlelength=1)
        ax.grid(True)
        
    for leftover_ax in axes[nplots:]:
        leftover_ax.axis("off")

    fig.suptitle("WEP Calculation Time for Different Representations", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_q_shor_vs_time_two_dfs(df1, df2, filepath):
    ncols = 2
    distances = sorted(set(df1["distance"].unique()).union(df2["distance"].unique()))
    nplots = len(distances)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()  # for easy indexing

    # Get all unique representations to assign colors consistently
    all_reps = sorted(set(df1["representation"].unique()).union(df2["representation"].unique()))
    palette = sns.color_palette("tab10", n_colors=len(all_reps))
    color_map = {rep: color for rep, color in zip(all_reps, palette)}

    for ax, distance_val in zip(axes, distances):
        # Prepare data for df1
        group1 = df1[df1["distance"] == distance_val]
        grouped1 = group1.groupby(["representation", 'q_shor'])['contraction_time'].mean().reset_index()
        pivot1 = grouped1.pivot(index='q_shor', columns='representation', values='contraction_time')

        # Prepare data for df2
        group2 = df2[df2["distance"] == distance_val]
        grouped2 = group2.groupby(["representation", 'q_shor'])['contraction_time'].mean().reset_index()
        pivot2 = grouped2.pivot(index='q_shor', columns='representation', values='contraction_time')

        # Plot df1 (solid lines)
        for rep in pivot1.columns:
            if(rep == "Concatenated"):
                ax.plot(pivot1.index, pivot1[rep], marker='o', linestyle='-', label=f"{rep} (custom)", color=color_map[rep])

        # Plot df2 (dotted lines)
        for rep in pivot2.columns:
            if(rep == "Concatenated"):
                ax.plot(pivot2.index, pivot2[rep], marker='o', linestyle=':', label=f"{rep} (default)", color=color_map[rep])

        ax.set_xlabel('q_shor')
        ax.set_ylabel('Average Time (s)')
        ax.set_title(f'Contraction Time vs. q_shor \n(Distance = {distance_val})')
        ax.legend(fontsize='x-small', markerscale=0.5, handlelength=1)
        ax.grid(True)

    for leftover_ax in axes[nplots:]:
        leftover_ax.axis("off")

    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()



def visualize_time_vs_metrics2(df1,
                              df2,
                              labels=("Size", "Combo"),
                              outpath="outputs/images/time_vs_metrics_operations.png"):

    metrics = [
        "contraction_cost",
        "avg_intermediate_tensor_size",
        "total_ops"
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(metrics),
        figsize=(5 * len(metrics), 4)
    )

    # Ensure axes is always iterable
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.scatter(df1[metric], df1["contraction_time"],
                   alpha=0.7, label=labels[0], color='blue', marker='o')
        ax.scatter(df2[metric], df2["contraction_time"],
                   alpha=0.7, label=labels[1], color='orange', marker='x')

        ax.set_xlabel(metric)
        ax.set_ylabel("contraction_time")
        ax.set_title(f"Time vs {metric}")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def custom_vs_default_cotengra_runtimes_boxplot(custom_df, default_df, output_path):
    default_df["source"] = "default_cotengra"
    custom_df["source"] = "custom"

    df_all = pd.concat([default_df, custom_df])
    df_all["code_type"] = df_all["q_shor"].astype(str) + " | " + df_all["representation"]

    distances = sorted(df_all["distance"].unique())
    n = len(distances)

    # Layout: try 2 columns, compute rows accordingly
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows), squeeze=False)
    fig.suptitle("Contraction Time per Code Type (Grouped by Distance)", fontsize=16)

    for idx, dist in enumerate(distances):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        df_dist = df_all[df_all["distance"] == dist]
        sns.boxplot(data=df_dist, x="code_type", y="contraction_time", hue="source", ax=ax)

        ax.set_title(f"Distance {dist}")
        ax.set_xlabel("Code Type (q_shor | Representation)")
        ax.set_ylabel("Contraction Time (s)")
        ax.tick_params(axis='x', rotation=45)
        ax.legend().set_title("Source")

    # Turn off any unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave room for suptitle
    plt.savefig(output_path)
    plt.close()


def make_table_default_vs_custom(df1, df2):
    df1 = df1[["distance", "q_shor", "representation", "contraction_time"]]
    df2 = df2[["distance","q_shor", "representation", "contraction_time"]]

    df1["source"] = "custom"
    df2["source"] = "default_cotengra"


    df_combined = pd.concat([df1, df2])
 
    # Group by q_shor, code_type, and source; then compute mean
    grouped = df_combined.groupby(["q_shor", "distance", "representation", "source"]).agg(
        avg_contraction_time=("contraction_time", "median")
    ).reset_index()

    # Optional: pivot to have "custom" and "default" side-by-side
    table = grouped.pivot(index=["distance", "q_shor", "representation"], columns="source", values="avg_contraction_time")

    # Optional: format nicely
    table = table.round(4).reset_index()
    return table


def visualize_time_vs_metrics3(df1,
                              df2,
                              outpath="outputs/images/time_vs_metrics_operations.png"):
    df1["source"] = "custom"
    df2["source"] = "default_cotengra"

    labels = ("custom", "default_cotengra")

    metrics = [
        "score_cotengra"
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(metrics),
        figsize=(5 * len(metrics), 4)
    )

    # Ensure axes is always iterable
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        ax.scatter(df1[metric], df1["contraction_time"],
                   alpha=0.7, label=labels[0], color='blue', marker='o')
        ax.scatter(df2[metric], df2["contraction_time"],
                   alpha=0.7, label=labels[1], color='orange', marker='x')

        ax.set_xlabel(metric)
        ax.set_ylabel("contraction_time")
        ax.set_ylim(0,200)
        ax.set_title(f"Time vs {metric}")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)

def plot_q_shor_vs_operations(df, filename, distance=None):
    if distance is not None:
        distances = [distance]
    else:
        distances = sorted(d for d in df["distance"].unique())
    ncols = 1
    nplots = len(distances)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    #axes = axes.flatten()  # for easy indexing
    ax = axes
    for distance_val in distances:
        group = df[df["distance"] == distance_val]
        grouped = group.groupby(["representation", 'q_shor']).agg(operations_mean=('operations', 'mean'), num_rows=('operations', 'size')).reset_index()
        representations = grouped["representation"].unique()
        print(grouped.head())

        for rep in representations:
            #ax.plot(pivot.index, pivot[rep], marker='o', label=rep)
            rep_data = grouped[grouped["representation"] == rep]
            x = rep_data["q_shor"]
            y = rep_data["operations_mean"]
            size = rep_data["num_rows"].min()
            ax.plot(x, y, marker='o', label=rep)

        ax.set_xlabel('p_flip (0.0 = surface, 1.0 = shor)')
        ax.set_ylabel('Operations')
        ax.set_yscale('log')
        ax.set_title(f'Distance = {distance_val}, Num Runs = {size}')
        ax.legend(fontsize='x-small', markerscale=0.5, handlelength=1)
        ax.grid(True)

    #fig.suptitle("WEP Calculation Time for Different Representations", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_cot_trials_vs_operations(df, filename):
    ncols = 2
    distances = sorted(d for d in df["distance"].unique())
    nplots = len(distances)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()  # for easy indexing

    for ax, distance_val in zip(axes, distances):
        group = df[df["distance"] == distance_val]
        grouped = group.groupby(["representation", 'max_repeats'])[['total_operations']].agg(['mean', 'std', 'size']).reset_index()
        representations = grouped["representation"].unique()

        for rep in representations:
            #ax.plot(pivot.index, pivot[rep], marker='o', label=rep)
            rep_data = grouped[grouped["representation"] == rep]
            print(rep_data)
            x = rep_data["max_repeats"]
            y = rep_data["total_operations"]["mean"]
            ax.plot(x, y, marker='o', label=rep)
            #yerr = rep_data["total_operations"]["std"]
    
            #ax.errorbar(x, y, yerr=yerr, marker='o', label=rep, capsize=5)

        ax.set_xlabel('Cotengra max repeats')
        ax.set_ylabel('Operations')
        ax.set_yscale('log')
        ax.set_title(f'Distance = {distance_val}, Runs = {rep_data["total_operations"]["size"].max()}')
        ax.legend(fontsize='x-small', markerscale=0.5, handlelength=1)
        ax.grid(True)
        
    for leftover_ax in axes[nplots:]:
        leftover_ax.axis("off")

    fig.suptitle("WEP Calculation Cost for different Cotengra Max Repeats", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_distance_vs_ops_for_qshor(df, filename, q_shor=None):
    ncols = 1
    q_shors = sorted(d for d in df["q_shor"].unique())
    if q_shor is not None:
        q_shors = [q_shor]

    nplots = len(q_shors)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    #axes = axes.flatten()  # for easy indexing

    # Filter data for the specified q_shor
    filtered = df[df['q_shor'] == q_shor]
    
    # Group by distance and representation, then average time
    grouped = filtered.groupby(['distance', 'representation'])['operations'].mean().reset_index()
    
    # Pivot to get representations as columns
    pivot = grouped.pivot(index='distance', columns='representation', values='operations')

    for rep in pivot.columns:
        axes.plot(pivot.index, pivot[rep], marker='o', label=rep)

    axes.set_xlabel('Distance')
    axes.set_ylabel('Total Operations - log scale')
    axes.set_yscale('log')
    axes.set_title(f'Operations vs. Distance \n(p_flip = {q_shor})')
    axes.legend()
    axes.grid(True)

    #fig.suptitle("Operations vs Distance for p_flip = 0.0", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_distance_vs_ops_upper(df, filename):
    # Group by distance and representation, then average time
    plt.figure(figsize=(10, 6))
    df['upper_bound_cost'] = df['upper_bound_cost'].astype('object').apply(int)

    grouped = df.groupby(["representation", 'distance'])[['upper_bound_cost']].agg(['mean', 'std', 'size']).reset_index()
    representations = grouped["representation"].unique()
    
    grouped2 = df.groupby(["representation", 'distance'])[['custom_cost']].agg(['mean', 'std', 'size']).reset_index()

    cmap = plt.get_cmap('tab10')  # or 'tab20', 'Set1', etc.
    color_dict = {rep: cmap(i % 10) for i, rep in enumerate(representations)}

    for rep in representations:
        rep_data = grouped[grouped["representation"] == rep]
        rep_data2 = grouped2[grouped2["representation"] == rep]

        color = color_dict[rep]

        #plt.plot(rep_data['distance'], rep_data['upper_bound_cost']['mean'], label=f"{rep}, runs: {rep_data['upper_bound_cost']['size'].max()}", marker='o', color=color)
        yerr = rep_data["upper_bound_cost"]["std"]
        #plt.errorbar(rep_data['distance'], rep_data['upper_bound_cost']['mean'], yerr=yerr, marker='o', label=f"{rep}, runs: {rep_data['upper_bound_cost']['size'].max()}", capsize=5, color=color)
        plt.plot(rep_data2['distance'], rep_data2['custom_cost']['mean'], marker='o', linestyle='--', alpha=0.5, color=color)

    #plt.yscale('log') 
    #plt.xscale('log')
    plt.xlabel('Distance')
    plt.ylabel('Upper Bound Cost')
    plt.title('Distance vs Upper Bound Cost by Name')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.close()


# Plotting distance vs operations for slurm in progress data
'''df = pd.read_csv('outputs/data/slurm_data_all_in_prog.csv', sep=';')
# Group by distance and representation, then average time
grouped = df.groupby(['q_shor', 'distance', 'representation'])['operations'].mean().reset_index()

df_intermediate = pd.read_csv("outputs/data/slurm_data_intermediate.csv", sep=';')
df_intermediate = df_intermediate.groupby(['q_shor', 'representation', 'distance'])['operations'].mean().reset_index()

merged = pd.concat([grouped, df_intermediate])
plot_distance_vs_ops_for_qshor(merged, "distance_vs_ops_log.png", q_shor=1.0)'''

'''df = pd.read_csv('custom_optimization_tests/combo_cotengra_with_many_max_repeats.csv', sep=';')
plot_cot_trials_vs_operations(df, "max_repeats_combo_d3.png")'''

df = pd.read_csv('upper_bound_and_custom_with_custom_cost.csv', sep=';')
#df2 = pd.read_csv('old_scripts/custom_cost_calc_ds.csv', sep=';')
#df2['custom_cost'] = np.log2(df2['custom_cost'].astype('object').apply(float))
plot_distance_vs_ops_upper(df, "distance_vs_ops_upper_bound_new.png")

#df = pd.read_csv('outputs/data/wep_calcs_full.csv', sep=';')
#plot_distance_vs_time_for_qshor(df, "distance_vs_time_for_surface_500_semilog.png", q_shor=0.0)

#df = pd.read_csv('contraction_time_vs_cost_info.csv', sep=';')
#visualize_time_vs_metrics(df, outpath="contraction_time_vs_metrics_logscale.png")



'''
df = pd.read_csv('outputs/combined_data/combined_data.csv', sep=';')
plot_q_shor_vs_operations(df, "p_flip_vs_operations.png", distance=3)'''
#plot_distance_vs_time_for_qshor(df)

# df1 = pd.read_csv("sparsity_tests_cotengra_size.csv", sep=';')
# df2 = pd.read_csv("sparsity_tests_cotengra_combo.csv", sep=';')
# visualize_time_vs_metrics2(df1, df2)


# custom_df = pd.read_csv("wep_calcs_custom_150.csv", sep=";")
# default_df = pd.read_csv("wep_calcs_default_150.csv", sep=";")
# print(make_table_default_vs_custom(custom_df, default_df))
#plot_q_shor_vs_time_two_dfs(custom_df, default_df, "outputs/images/custom_vs_default_concatenated.png")
#visualize_time_vs_metrics3(custom_df, default_df, outpath="outputs/images/time_vs_metrics_operations_custom_vs_default.png")


#df = pd.read_csv("optimization_tests_with_legs_cotengra.csv", sep=";")
#visualize_time_vs_metrics(df, outpath="outputs/images/contraction_time_vs_metrics_default_cotengra.png")