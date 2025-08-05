import pandas as pd

# df_default_ctg = pd.read_csv("outputs/open_legs_tests/default_ctg_combined_cost.csv", sep=';')
# df_default_ctg = df_default_ctg[["distance", "q_shor", "max_repeats", "representation", "contraction_time", "total_ops", "score_cotengra"]]
# df_default_ctg = df_default_ctg.rename(columns={
#     "total_ops": "operations"
# })

df_default_slurm_d3 = pd.read_csv("outputs/slurm_data/custom_ctg/d3_info.csv", sep=';')
print(df_default_slurm_d3.head())
df_default_slurm_d3 = df_default_slurm_d3[["distance", "q_shor", "representation", "contraction_time", "operations", "score_cotengra"]]
print(df_default_slurm_d3.head())
# df_default_slurm_d4 = pd.read_csv("outputs/slurm_data/d4_info.csv", sep=';')
# df_default_slurm_d4 = df_default_slurm_d4[["distance", "q_shor", "representation", "contraction_time", "operations"]]

df_default_slurm_d5 = pd.read_csv("outputs/slurm_data/custom_ctg/d5_info.csv", sep=';')
df_default_slurm_d5 = df_default_slurm_d5[["distance", "q_shor", "representation", "contraction_time", "operations", "score_cotengra"]]

df_default_slurm_d7 = pd.read_csv("outputs/slurm_data/custom_ctg/d7_info.csv", sep=';')
df_default_slurm_d7 = df_default_slurm_d7[["distance", "q_shor", "representation", "contraction_time", "operations", "score_cotengra"]]

# df_default_slurm_d9 = pd.read_csv("outputs/slurm_data/d9_info.csv", sep=';')
# df_default_slurm_d9 = df_default_slurm_d9[["distance", "q_shor", "representation", "contraction_time", "operations"]]

df_default_slurm = pd.concat([
    df_default_slurm_d3,
    #df_default_slurm_d4,
    df_default_slurm_d5,
    df_default_slurm_d7,
    #df_default_slurm_d9
], ignore_index=True)
df_default_slurm["max_repeats"] = 300

df_default_slurm.to_csv("outputs/combined_data/combined_data_custom.csv", sep=';', index=False)