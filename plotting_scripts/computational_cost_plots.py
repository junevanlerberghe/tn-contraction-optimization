# Group by distance and representation, then average time
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

n_vals = np.linspace(9, 400, 300)
alpha = 2
D = 2

def tree(n): return np.log(n)
def tree_1d(n): return n
def hyperbolic(n): return n**(alpha + 1)
def hypercubic(n): return n*np.exp(n**(1-1/D))
def hypercubic_l(n): 
    L = (-1 + np.sqrt(1 + 4*n))/2
    return n*np.exp(L**(D-1))
def volume_law(n):
    d = 0.5
    return n*np.exp(d*n)
def generic(n):
    return (n**2)*np.exp(n)/np.log(n)
    


plt.figure(figsize=(10, 6))
plt.plot(n_vals, np.log(tree(n_vals)), label='tree', color='blue')
plt.plot(n_vals, np.log(tree_1d(n_vals)), label='Tree, 1d', color='purple')
plt.plot(n_vals, np.log(hyperbolic(n_vals)), label='2d hyperbolic', color='red')
plt.plot(n_vals, np.log(hypercubic(n_vals)), label='hypercubic', color='orange')
plt.plot(n_vals, np.log(hypercubic_l(n_vals)), label='hypercubic (bounded L)', color='green')
plt.plot(n_vals, np.log(volume_law(n_vals)), label='volume law')
plt.plot(n_vals, np.log(generic(n_vals)), label='generic', color='black')


df = pd.read_csv("tn_architectures_costs.csv", sep=';')
df['upper_bound_cost'] = df['upper_bound_cost'].astype('object').apply(int)

grouped = df.groupby(["representation", 'num_qubits'])[['upper_bound_cost']].agg(['mean', 'std', 'size']).reset_index()
representations = grouped["representation"].unique()
#df["custom_cost"] = np.exp(df["custom_cost"])
grouped2 = df.groupby(["representation", 'num_qubits'])[['custom_cost']].agg(['mean', 'std', 'size']).reset_index()

manual_color_label_map = {
    'Concatenated': ('blue', 'Concatenated'),
    'Holographic': ('red', 'Holographic'),
    'Rotated Surface': ('orange', 'Rotated Surface'),
    'Rectangular Surface': ('green', 'Rectangular Surface'),
    'Hamming MSP (non-degenerate)': ('brown', 'Hamming MSP (non-degenerate)'),
    'BB MSP (degenerate)': ('black', 'BB MSP (degenerate)'),
}


for rep in representations:
    rep_data = grouped[grouped["representation"] == rep]
    rep_data2 = grouped2[grouped2["representation"] == rep]

    color, display_label = manual_color_label_map.get(rep, ('gray', rep))  # fallback color and label
    plt.errorbar(
        rep_data2["num_qubits"],
        rep_data2['custom_cost']['mean'],
        yerr=rep_data2['custom_cost']['std'],  # std deviation as error bar length
        fmt='o',              # marker style
        linestyle='--',
        alpha=0.5,
        color=color,
        label=display_label,
        capsize=3            # adds little horizontal bars at error bar ends
    )

##plt.yscale('log') 
plt.xscale('log')
plt.xlim(0,500)
plt.ylim(0,30)
plt.xlabel('n qubits')
plt.ylabel('Computational Cost')
plt.title('Computational Cost by TN Architecture')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('tn_architectures_zoomed.png', dpi=300)
plt.close()
