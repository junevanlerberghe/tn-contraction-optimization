import math
import pandas as pd
import matplotlib.pyplot as plt


def plot_contraction_steps_for_representations():
    df = pd.read_csv("contraction_steps_with_open_legs.csv", sep=';')
    df['time_step'] = df.groupby(['num_run', 'representation']).cumcount()

    # Filter columns of interest
    columns_to_plot = ['time_step', 'representation', 'cost', 'pte1 sparsity', 'pte2 sparsity']
    df_plot = df[columns_to_plot]

    # Get unique representations
    representations = df_plot['representation'].unique()

    num_reps = len(representations)

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8), sharex=False)
    axes = axes.flatten()

    # Plot each representation's raw cost data
    for ax, rep in zip(axes, representations):
        rep_df = df_plot[df_plot['representation'] == rep]
        #ax.plot(rep_df['time_step'], rep_df['cost'], label='Cost', marker='o')
        ax.plot(rep_df['time_step'], rep_df['pte1 sparsity'], label='pte1', marker='x')
        ax.plot(rep_df['time_step'], rep_df['pte2 sparsity'], label='pte2', marker='^')
        ax.set_title(f"Representation: {rep}")
        ax.set_ylabel("PTE Sparsity")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time Step")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("contraction_steps_with_open_legs.png")
    plt.close()


def plot_contraction_steps_for_runs():
    df = pd.read_csv("contraction_steps/contraction_steps_tanner_d4.csv", sep=';')
    df_slowest = df[df["num_run"] == 3].reset_index(drop=True)  
    df_fastest = df[df["num_run"] == 0].reset_index(drop=True)

    plt.figure(figsize=(10, 6))

    plt.plot(df_slowest.index, df_slowest['cost'], label='Slowest', marker='o')
    plt.plot(df_fastest.index, df_fastest['cost'], label='Fastest', marker='s')

    plt.xlabel('Step')
    plt.ylabel('Cost')
    plt.title('Contraction Cost per Step')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("contraction_steps_d4_comparison.png")
    plt.close()
    

plot_contraction_steps_for_representations()

