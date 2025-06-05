import math
import pandas as pd
import matplotlib.pyplot as plt

def visualize_time_vs_metrics(df,
                              outpath="outputs/images/time_vs_metrics_operations.png"):

    metrics = [
        "contraction_cost",
        "avg_intermediate_tensor_size",
        "total_operations"
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(metrics),
        figsize=(5 * len(metrics), 4)
    )

    for ax, metric in zip(axes, metrics):
        ax.scatter(df[metric], df["contraction_time"], alpha=0.7)
        ax.set_xlabel(metric)
        ax.set_ylabel("contraction_time")
        ax.set_title(f"Time vs {metric}")
        ax.grid(True)

    plt.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_distance_vs_time_for_qshor(df):
    ncols = 2
    q_shors = sorted(d for d in df["q_shor"].unique())
    nplots = len(q_shors)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()  # for easy indexing

    for ax, q_shor in zip(axes, q_shors):
        # Filter data for the specified q_shor
        filtered = df[df['q_shor'] == q_shor]
        
        # Group by distance and representation, then average time
        grouped = filtered.groupby(['distance', 'representation'])['time'].mean().reset_index()
        
        # Pivot to get representations as columns
        pivot = grouped.pivot(index='distance', columns='representation', values='time')

        for rep in pivot.columns:
            ax.plot(pivot.index, pivot[rep], marker='o', label=rep)

        ax.set_xlabel('Distance')
        ax.set_ylabel('Average Time (s)')
        ax.set_title(f'Contraction Time vs. Distance \n(q_shor = {q_shor})')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("outputs/images/distance_vs_time.png", dpi=300)
    plt.close()

        

def plot_q_shor_vs_time(df):
    ncols = 2
    distances = sorted(d for d in df["distance"].unique())
    nplots = len(distances)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten()  # for easy indexing

    for ax, distance_val in zip(axes, distances):
        group = df[df["distance"] == distance_val]
        grouped = group.groupby(["representation", 'q_shor'])['time'].mean().reset_index()
        pivot = grouped.pivot(index='q_shor', columns='representation', values='time')

        for rep in pivot.columns:
            ax.plot(pivot.index, pivot[rep], marker='o', label=rep)

        ax.set_xlabel('q_shor')
        ax.set_ylabel('Average Time (s)')
        ax.set_title(f'Contraction Time vs. q_shor \n(Distance = {distance_val})')
        ax.legend()
        ax.grid(True)
        
    for leftover_ax in axes[nplots:]:
        leftover_ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/images/q_shor_vs_time.png", dpi=300)
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



#df = pd.read_csv('wep_calcs.csv', sep=';')
#plot_q_shor_vs_time(df)
#plot_distance_vs_time_for_qshor(df)

# df1 = pd.read_csv("sparsity_tests_cotengra_size.csv", sep=';')
# df2 = pd.read_csv("sparsity_tests_cotengra_combo.csv", sep=';')
# visualize_time_vs_metrics2(df1, df2)


df = pd.read_csv("sparsity_tests.csv", sep=";")
visualize_time_vs_metrics(df)