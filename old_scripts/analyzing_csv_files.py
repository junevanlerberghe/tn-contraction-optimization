from matplotlib import pyplot as plt
import pandas as pd
from tabulate import tabulate
import seaborn as sns


filename = 'pcm_tests_data/pcm_sizes_comparison_d4_rotated_surface_0.5.csv'
#df_rotated = pd.read_csv(filename, sep=';')  # use sep if needed
#df_concat = pd.read_csv("pcm_tests_data/pcm_sizes_comparison_d4_concatenated_0.5.csv", sep=';')
#df_dual = pd.read_csv("pcm_tests_data/pcm_sizes_comparison_d4_dual_surface_0.5.csv", sep=';')
#df_msp = pd.read_csv("pcm_tests_data/pcm_sizes_comparison_d3_msp_0.5.csv", sep=';')
df_tanner = pd.read_csv("pcm_tests_data/pcm_sizes_comparison_d3_tanner_0.5.csv", sep=';')
#df = pd.concat([df_rotated, df_concat, df_dual, df_msp, df_tanner], ignore_index=True)
#print(tabulate(df, headers='keys', tablefmt='pretty'))
df = df_tanner
df_filtered = df[df['self trace?'] != True]
df_filtered[['new_pcm_rows', 'new_pcm_cols']] = df_filtered['new pcm size'].str.extract(r'\((\d+),\s*(\d+)\)').astype(int)
df_filtered[['pcm_1_rows', 'pcm_1_cols']] = df_filtered['pcm_1 size'].str.extract(r'\((\d+),\s*(\d+)\)').astype(int)
df_filtered[['pcm_2_rows', 'pcm_2_cols']] = df_filtered['pcm_2 size'].str.extract(r'\((\d+),\s*(\d+)\)').astype(int)

df_filtered['pcm_1_total_size'] = df_filtered['pcm_1_rows'] * df_filtered['pcm_1_cols']
df_filtered['pcm_2_total_size'] = df_filtered['pcm_2_rows'] * df_filtered['pcm_2_cols']
df_filtered['new_pcm_total_size'] = df_filtered['new_pcm_rows'] * df_filtered['new_pcm_cols']

fig, axs = plt.subplots(2, 4, figsize=(16, 10))

# Plot 1: New PCM Total Size vs Cost
axs[0,0].scatter(df_filtered['new_pcm_total_size'], df_filtered['cost'], alpha=0.7)
axs[0, 0].set_xlabel('New PCM Total Size (rows × cols)')
axs[0, 0].set_ylabel('Cost')
#axs[0, 0].set_ylim(0, 50000)
axs[0, 0].set_title('Cost vs. New PCM Size')
axs[0, 0].grid(True)

# Plot 2: New PCM Sparsity vs Cost
axs[0, 1].scatter(df_filtered['new pcm sparsity'], df_filtered['cost'], alpha=0.7)
axs[0, 1].set_xlabel('New PCM Sparsity')
axs[0, 1].set_ylabel('Cost')
#axs[0, 1].set_ylim(0, 50000)
axs[0, 1].set_title('Cost vs. New PCM Sparsity')
axs[0, 1].grid(True)

# Plot 3: PCM 1 Size vs PTE 1 Length
axs[1, 0].scatter(df_filtered['pcm_1_total_size'], df_filtered['pte 1 len'], alpha=0.7)
axs[1, 0].set_xlabel('PCM 1 Total Size (rows × cols)')
axs[1, 0].set_ylabel('PTE 1 Length')
#axs[1, 0].set_ylim(0, 1200)
axs[1, 0].set_title('PTE 1 Length vs. PCM 1 Size')
axs[1, 0].grid(True)

# Plot 4: PCM 2 Size vs PTE 2 Length
axs[1, 1].scatter(df_filtered['pcm_2_total_size'], df_filtered['pte 2 len'], alpha=0.7)
axs[1, 1].set_xlabel('PCM 2 Total Size (rows × cols)')
axs[1, 1].set_ylabel('PTE 2 Length')
#axs[1, 1].set_ylim(0, 1200)
axs[1, 1].set_title('PTE 2 Length vs. PCM 2 Size')
axs[1, 1].grid(True)

# Plot 5: PTE Lengths vs Cost
axs[0, 2].scatter(df_filtered['pte 1 len']*df_filtered['pte 2 len']*0.25, df_filtered['cost'], alpha=0.7)
axs[0, 2].set_xlabel('PTE 1 Len * PTE 2 Len')
axs[0, 2].set_ylabel('Cost')
#axs[0, 2].set_ylim(0, 10000)
#axs[0, 2].set_xlim(0, 10000)
axs[0, 2].set_title('Cost vs. PTE Lengths')
axs[0, 2].grid(True)

# # Plot 6: PCM Sparsity vs PTE Length
axs[1, 2].scatter(df_filtered['pcm_1 sparsity'], df_filtered['pte 1 len'], alpha=0.7)
axs[1, 2].set_xlabel('PCM 1 Sparsity')
axs[1, 2].set_ylabel('PTE 1 Length')
axs[1, 2].set_title('PCM 1 Sparsity vs PTE 1 Length')
axs[1, 2].grid(True)

# Plot 7: PCM Sparsity vs PTE Length
axs[0, 3].scatter(df_filtered['pcm_2 sparsity'], df_filtered['pte 2 len'], alpha=0.7)
axs[0, 3].set_xlabel('PCM 2 Sparsity')
axs[0, 3].set_ylabel('PTE 2 Length')
axs[0, 3].set_title('PCM 2 Sparsity vs PTE 2 Length')
axs[0, 3].grid(True)


plt.tight_layout()
plt.savefig("pcm_tests_data/pcm_cost_tanner_d4_0.5.png")
plt.close()

# # Plot rows vs cost
# plt.subplot(2, 2, 2)
# plt.scatter(df_filtered['new_pcm_rows'], df_filtered['cost'], alpha=0.7)
# plt.xlabel('New PCM Rows')
# plt.ylabel('Cost')
# plt.title('Cost vs. New PCM Rows')
# plt.grid(True)

# # Plot cols vs cost
# plt.subplot(2, 2, 3)
# plt.scatter(df_filtered['new_pcm_cols'], df_filtered['cost'], alpha=0.7)
# plt.xlabel('New PCM Cols')
# plt.ylabel('Cost')
# plt.title('Cost vs. New PCM Cols')
# plt.grid(True)



