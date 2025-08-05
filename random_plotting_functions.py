from collections import defaultdict
import math
import time
from typing import List, Tuple
from galois import GF2
from matplotlib import pyplot as plt
from scipy.stats import linregress

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tnqec'))

import numpy as np
import pandas as pd

#from wep_calculations import stabilizer_enumerator_polynomial


#coloring = [[2, 2, 2, 2, 2], [2, 1, 2, 2, 2], [1, 2, 1, 1, 2], [2, 2, 2, 2, 1], [2, 2, 2, 2, 2]]
#coloring = [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
#compass = CompassCode(4, coloring)
#tn = compass.rotated()
#wep = tn.stabilizer_enumerator_polynomial(
#        verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True
#    )
#print(wep)


'''df = pd.read_csv("optimization_tests_with_legs_cotengra.csv", sep=";")
df2 = pd.read_csv("optimization_tests_with_custom_func.csv", sep=";")

avg_contraction_time = df["contraction_time"].mean()
avg_contraction_time2 = df2["contraction_time"].mean()
print("Average contraction time with default cotengra:", avg_contraction_time)
print("Average contraction time with custom function:", avg_contraction_time2)'''


'''def generate_checkerboard_coloring(d):
    return [[1 + (i + j) % 2 for j in range(d-1)] for i in range(d-1)]

ds = [3]
q_shors = [1.0]

for d in ds:
    for q_shor in q_shors:
        # if random number is less than q_shor, then put an X stabilizer in that plaquette
        # start with distance d checkerboard coloring, then change to Xs whenver necessary
        coloring = generate_checkerboard_coloring(d)
        for i in range(d - 1):
            for j in range(d - 1):
                if np.random.rand() < q_shor:
                    coloring[i][j] = 2

coloring = [[2,2], [2,2]]
compass = CompassCode(3, coloring)
tn = compass.concatenated()
print(tn)
wep, contraction_width, contraction_cost, intermediate_tensor_sizes, total_ops_count = stabilizer_enumerator_polynomial(tn, verbose=False, progress_reporter=TqdmProgressReporter(), cotengra=True)
print(tn.conjoin_nodes().h)
print(wep)'''

# FIND AVERAGES OF DEFAULT COTENGRA VS CUSTOM SCORES
df_default = pd.read_csv("outputs/combined_data/combined_data.csv", sep=";")
df_custom = pd.read_csv("outputs/combined_data/combined_data_custom.csv", sep=";")

def prepare_data(df_default, df_custom, distance):
    df_default_filtered = df_default[(df_default["q_shor"] == 0.0) & (df_default["distance"] == distance)]
    df_custom_filtered = df_custom[(df_custom["q_shor"] == 0.0) & (df_custom["distance"] == distance)]

    default_stats = (
        df_default_filtered.groupby("representation")["operations"]
        .agg(default_avg_ops='mean', default_std_ops='std', default_num_rows='count')
        .reset_index()
    )

    custom_stats = (
        df_custom_filtered.groupby("representation")["operations"]
        .agg(custom_avg_ops='mean', custom_std_ops='std', custom_num_rows='count')
        .reset_index()
    )

    final_df = pd.merge(default_stats, custom_stats, on="representation")
    final_df["percentage decrease (not log)"] = (final_df["default_avg_ops"] - final_df["custom_avg_ops"]) / final_df["default_avg_ops"]
    final_df = final_df.sort_values("default_avg_ops", ascending=False)
    return final_df

final_dfs = {
    "d3": prepare_data(df_default, df_custom, 3),
    "d5": prepare_data(df_default, df_custom, 5),
    "d7": prepare_data(df_default, df_custom, 7),
}

print(final_dfs)
# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

for ax, (dist, final_df) in zip(axes, final_dfs.items()):
    x = np.arange(len(final_df))
    width = 0.35

    ax.bar(x - width/2, final_df["default_avg_ops"], width, yerr=final_df["default_std_ops"],
           capsize=4, label=f"Default - # runs: {final_df['default_num_rows'].max()}", color="skyblue", edgecolor="black")
    ax.bar(x + width/2, final_df["custom_avg_ops"], width, yerr=final_df["custom_std_ops"],
           capsize=4, label=f"Custom - # runs: {final_df['custom_num_rows'].max()}", color="lightgreen", edgecolor="black")
    
    ax.set_xticks(x)
    ax.set_yscale("log")
    ax.set_xticklabels(final_df["representation"], rotation=45, ha='right')
    ax.set_title(f"Contraction Cost Comparison ({dist})")
    ax.set_ylabel("log₂(Total Ops)")
    ax.legend()

plt.tight_layout()
plt.suptitle("Default vs Custom Contraction Costs", y=1.02, fontsize=16)
plt.savefig("bar_chart_default_vs_custom.png")
plt.close()

# same for d4
# df_default = pd.read_csv("outputs/data/slurm_data_d4_intermediate.csv", sep=";")
# df_custom = pd.read_csv("custom_optimization_tests/time_vs_cost_pcm_avg_size_model_fit.csv", sep=";")

# df_default = df_default[(df_default["q_shor"] == 0.0) & (df_default["distance"] == 4)]
# #df_default["operations"] = np.log2(df_default["operations"])
# avg_default = df_default.groupby("representation")["operations"].mean().rename("default_avg")
# std_default = df_default.groupby("representation")["operations"].std().rename("default_std")
# df_default_combined = pd.concat([avg_default, std_default], axis=1)

# df_custom = df_custom[(df_custom["q_shor"] == 0.0) & (df_custom["distance"] == 4)]
# #df_custom["total_operations"] = np.log2(df_custom["total_operations"])
# avg2 = df_custom.groupby("representation")["total_operations"].mean().rename("pcm_size_avg")
# std2 = df_custom.groupby("representation")["total_operations"].std().rename("pcm_size_std")
# df_combined = pd.concat([avg2, std2], axis=1)

# final_df = pd.merge(df_default_combined, df_combined, on="representation")

# print("Combined contraction cost comparison for d4 (default = 50 runs, custom = 34 runs):")
# print(final_df)

# plot contraction time vs average PCM size
'''df = pd.read_csv("time_vs_pcm_sizes/tanner_network.csv", sep=";")

plt.scatter(df["avg rows"], df["contraction_time"], marker='o')

plt.xlabel('Average PCM Size')
plt.ylabel('Contraction Time')
plt.title('Contraction Time vs Average PCM Size')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'''

# Cotengra score vs my custom scores against contraction time
'''fig, axs = plt.subplots(1, 2, figsize=(12, 8))
axs = axs.flatten()

df = pd.read_csv("outputs/combined_data/submatrix_rank.csv", sep=";")
#df = df[df["representation"] == "Concatenated"]

axs[0].scatter((df["score cotengra"]), df["total_ops_count"], marker='o')
axs[0].set_xlabel('Cotengra Score')
axs[0].set_ylabel('Contraction Cost (Operations)')
axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].set_title('Contraction Cost vs Cotengra Score')
axs[0].grid(True)

axs[1].scatter((df["total_submatrix_rank_cost"]), df["total_ops_count"], marker='o')
axs[1].set_xlabel('Submatrix Rank Cost')
axs[1].set_ylabel('Contraction Cost (Operations)')
axs[1].set_title('Contraction Cost vs Submatrix Rank Cost')
axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[1].grid(True)

plt.tight_layout()
plt.suptitle("Cotengra vs Custom Score. d = 3, 4. runs = 100", y=1.02, fontsize=16)
plt.savefig("cotengra_vs_custom_scatter.png")
plt.close()'''

# Plot representations as different colors:
fig, axs = plt.subplots(1, 2, figsize=(12, 8))
axs = axs.flatten()

df = pd.read_csv("outputs/combined_data/combined_data.csv", sep=";")
df_custom = pd.read_csv("outputs/combined_data/combined_data_custom.csv", sep=";")
#df["Open Legs Cost"] = 0.028*(df["Open Legs Cost"]**2.26)
#df = df[df["distance"] == 5]
#df = df[df["representation"] == "Concatenated"]  

representations = df["representation"].unique()
colors = plt.cm.tab10.colors  # or use 'Set3', 'Accent', etc.

# Map each representation to a color
color_map = {rep: colors[i % len(colors)] for i, rep in enumerate(representations)}

# Plotting function
def plot_by_rep(df, x_col, ax, title):
    for rep in representations:
        subdf = df[df["representation"] == rep]
        ax.scatter(subdf[x_col], subdf["operations"], label=rep, color=color_map[rep])
    ax.set_xlabel(x_col)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Operations")
    ax.set_title(title)
    #ax.legend()
    ax.grid(True)

# Create the four plots
plot_by_rep(df, "score_cotengra", axs[1], "Operations vs Cotengra Score")
plot_by_rep(df_custom, "score_cotengra", axs[0], "Operations vs Custom Score")


# ax = axs[7]
# for rep in representations:
#     subdf = df[df["representation"] == rep]
#     ax.scatter(subdf["pcm cost"] * subdf["non_mixed_cols"], subdf["total_ops_count"], label=rep, color=color_map[rep])
# ax.set_xlabel("pcm cost * non_mixed_cols")
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_ylabel("Operations")
# ax.set_title("Operations vs non_mixed_cols * PCM Cost")
# ax.grid(True)

handles, labels = axs[1].get_legend_handles_labels()

# Add one legend for the whole figure
fig.legend(handles, labels, loc='upper left', ncol=1)

fig.suptitle("Cost vs Scores", fontsize=16)
plt.tight_layout()
plt.savefig("custom_scatter_results.png")
plt.close()

# Plot cost vs open legs per contraction step
'''
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs = axs.flatten()

df = pd.read_csv("new_length_estimate_calcs/contraction_steps_submatrix.csv", sep=";")

df.loc[df["pte1 sparsity"] > 1, "pte1 sparsity"] = 1
df.loc[df["pte2 sparsity"] > 1, "pte2 sparsity"] = 1

df["min_open_legs"] = df[["open_legs1", "open_legs2"]].min(axis=1)
print(df.head())

df["new cost"] = np.where(df["self trace?"] == True, 
                          df["open_legs1"],
                          df["open_legs1"] + df["open_legs2"])

heuristic = np.sqrt((df["open_legs1"] + df["open_legs2"])*(1-((df["pte1 sparsity"] + df["pte2 sparsity"])/2)))
wep_paper = (df["open_legs1"] + df["open_legs2"] + df["min_open_legs"])
lengths = df["pte 1 len"] * df["pte 2 len"]
combined = np.sqrt(wep_paper * df["pcm size cost"])

axs[0].scatter(wep_paper, df["cost"], marker='o')
axs[0].set_xlabel('open legs m, n: m + n + min(m, n)')
axs[0].set_yscale('log')
axs[0].set_xscale('log')
#axs[0].set_xlim(0, 75)
#axs[0].set_ylim(0, 600)
axs[0].set_ylabel('Cost')
axs[0].set_title('Cost vs Upper Bound on Complexity (log-log)')
axs[0].grid(True)

axs[1].scatter(df["custom_cost"], df["cost"], marker='o')
axs[1].set_xlabel('cost based on PTE lengths')
axs[1].set_yscale('log')
axs[1].set_xscale('log')
#axs[0].set_xlim(0, 75)
#axs[0].set_ylim(0, 600)
axs[1].set_ylabel('Cost')
axs[1].set_title('Cost vs PTE length estimate')
axs[1].grid(True)


axs[2].scatter((lengths), df["cost"], marker='o')
axs[2].set_xlabel('len(pte1) * len(pte2)')
axs[2].set_yscale('log')
axs[2].set_xscale('log')
#axs[1].set_xlim(0, 75)
#axs[1].set_ylim(0, 600)
axs[2].set_ylabel('Cost')
axs[2].set_title('Cost vs PTE Lengths')
axs[2].grid(True)


df[['rows', 'cols']] = df['new pcm size'].str.extract(r'\((\d+),\s*(\d+)\)').astype(int)

df["new_pcm_cost_2"] = 2**(np.log2(df["rows"] * df["cols"])) / 2

axs[3].scatter(df["pcm size cost"], df["cost"], marker='o')
axs[3].set_xlabel('pcm size cost')
axs[3].set_yscale('log')
axs[3].set_xscale('log')
axs[3].set_ylabel('Cost')
axs[3].set_title('Cost vs PCM size')
axs[3].grid(True)

# from sklearn.tree import DecisionTreeClassifier

# features = ["rows", "cols", "self trace?", "open_legs1", "open_legs2"]
# X = df[features].fillna(0)
# y = df["cost"]

# model = DecisionTreeClassifier(max_depth=5)
# model.fit(X, y)
# df["cost_tree"] = model.predict(X)

df["new_pcm_cost_2"] = 2**(np.log2(df["rows"] * df["cols"])) / 2

df["new_pcm_cost_w_self"] = np.where(df["self trace?"] == True, 
                          2**(np.log2((df["rows"] - 1) * (df["cols"] - 4) * (df["open_legs1"]))),
                          2**(np.log2(df["rows"] * df["cols"] * (df["open_legs1"] + df["open_legs2"] + df["min_open_legs"]))))



axs[4].scatter(df["new_pcm_cost_w_self"], df["cost"], marker='o')
axs[4].set_yscale('log')
axs[4].set_xscale('log')
axs[4].set_xlabel('pcm size cost w self trace')
axs[4].set_ylabel('Cost')
axs[4].set_title('Cost vs PCM Size Cost with self tracing')
axs[4].grid(True)


axs[5].scatter(df["rank_submatrix"], df["cost"], marker='o')
axs[5].set_yscale('log')
axs[5].set_xscale('log')
axs[5].set_xlabel('rank_submatrix')
axs[5].set_ylabel('Cost')
axs[5].set_title('Cost vs rank_submatrix')
axs[5].grid(True)

plt.tight_layout()
plt.savefig("cost_per_contraction_step_submatrix.png")
plt.close()'''

# #x = df["rows"] + df["cols"] + np.minimum(df["rows"], df["cols"])
# y = df["cost"]
# x = df["new_pcm_cost_w_self"]*df["custom_cost"]
# # Filter to remove non-positive values
# mask = (x > 0) & (y > 0)
# x = x[mask]
# y = y[mask]

# # Take log10
# log_x = np.log10(x)
# log_y = np.log10(y)

# # fit line
# slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)

# C = 10 ** intercept
# k = slope

# print(f"Estimated cost ≈ {C:.3f} * (score)^{k:.2f}")


# Finding minimum operations for each representation given a df
'''df = pd.read_csv("custom_optimization_tests/combo_cotengra_with_many_max_repeats.csv", sep=";")
min_ops_per_representation = df.loc[df.groupby("representation")["total_operations"].idxmax()]
print("Maximum operations per representation:")
print(min_ops_per_representation[["representation", "total_operations"]])'''

# Finding average operations count for default vs custom cotengra:
'''df = pd.read_csv('new_subrank_cost_results.csv', sep=';')
df = df.groupby(['q_shor', 'distance', 'representation'])['total_ops'].std().reset_index()
df['total_ops'] = np.log2(df['total_ops'])

print(df[['q_shor', 'distance', 'representation', 'total_ops']])

df_custom = pd.read_csv('custom_optimization_tests/custom_ctg_d4.csv', sep=';')
df_custom = df_custom.groupby(['q_shor', 'distance', 'representation'])['total_ops'].std().reset_index()
df_custom['total_ops'] = np.log2(df_custom['total_ops'])
print(df_custom[['q_shor', 'distance', 'representation', 'total_ops']])'''


'''df = pd.read_csv("steps_for_new_self_trace.csv", sep=";")
filtered_df = df[df['cost'] != df['new_subrank_cost']]
filtered_df.to_csv('cost_mismatch_rows_length.csv', index=False, sep= ';')'''



'''fig, axs = plt.subplots(1, 3, figsize=(14, 8))
axs = axs.flatten()

df = pd.read_csv("outputs/results_7_25/new_submatrix_rank_calc.csv", sep=";")
#df = df[df["representation"] == "Concatenated"]

axs[0].scatter((df["score_cotengra"]), df["total_ops_count"], marker='o')
axs[0].set_xlabel('Cotengra Score')
axs[0].set_ylabel('Contraction Cost (Operations)')
axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].set_title('Contraction Cost vs Cotengra Score')
axs[0].grid(True)

axs[1].scatter((df["custom_subrank_cost"]), df["total_ops_count"], marker='o')
axs[1].set_xlabel('Custom Cost')
axs[1].set_ylabel('Contraction Cost (Operations)')
axs[1].set_title('Contraction Cost vs Custom Cost')
axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[1].grid(True)


# axs[2].scatter(df["subrank_w_self_trace"], df["total_ops_count"], marker='o', color=np.where(df["subrank_w_self_trace"] < df["total_ops_count"], "red", "blue"))
# axs[2].plot(df["subrank_w_self_trace"],df["subrank_w_self_trace"])
# axs[2].set_xlabel('Previous Submatrix Rank Cost')
# axs[2].set_ylabel('Contraction Cost (Operations)')
# axs[2].set_title('Contraction Cost vs Prev Submatrix Rank Cost')
# # axs[2].set_yscale('log')
# axs[2].set_xscale('log')
# axs[2].grid(True)

plt.tight_layout()
plt.suptitle("Cotengra vs Custom Scores with self trace", y=1.02, fontsize=16)
plt.savefig("self_trace_scatter_new.png")
plt.close()'''



fig, axs = plt.subplots(1, 3, figsize=(14, 8))
axs = axs.flatten()

df = pd.read_csv("outputs/combined_data/combined_data.csv", sep=";")
#df = df[df["representation"] == "Concatenated"]

axs[0].scatter((df["score_cotengra"]), df["operations"], marker='o')
axs[0].set_xlabel('Cotengra Score')
axs[0].set_ylabel('Contraction Cost (Operations)')
axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].set_title('Contraction Cost vs Cotengra Score')
axs[0].grid(True)

df = pd.read_csv("outputs/combined_data/combined_data_custom.csv", sep=";")
axs[1].scatter((df["score_cotengra"]), df["operations"], marker='o')
axs[1].set_xlabel('Custom Cost')
axs[1].set_ylabel('Contraction Cost (Operations)')
axs[1].set_title('Contraction Cost vs Custom Cost')
axs[1].set_yscale('log')
axs[1].set_xscale('log')
axs[1].grid(True)


# axs[2].scatter(df["subrank_w_self_trace"], df["total_ops_count"], marker='o', color=np.where(df["subrank_w_self_trace"] < df["total_ops_count"], "red", "blue"))
# axs[2].plot(df["subrank_w_self_trace"],df["subrank_w_self_trace"])
# axs[2].set_xlabel('Previous Submatrix Rank Cost')
# axs[2].set_ylabel('Contraction Cost (Operations)')
# axs[2].set_title('Contraction Cost vs Prev Submatrix Rank Cost')
# # axs[2].set_yscale('log')
# axs[2].set_xscale('log')
# axs[2].grid(True)

plt.tight_layout()
plt.suptitle("Cotengra vs Custom Scores with self trace", y=1.02, fontsize=16)
plt.savefig("self_trace_scatter_new.png")
plt.close()