# Hyperoptimized Quantum Lego Contraction Schedules

This repository incudes the code and data needed to reproduce all data and plots for the paper "Hyperoptimized Quantum Lego Contraction Schedules" by Balint Pato, June Vanlerberghe, and Kenneth R. Brown (2025). 

## Data and Images

All data and plots used in the paper are located in the `results` directory. Generate plots are in `results\images` and the data used to generate the plots is in `results/data`.

## Setup Instructions

To set up the environment and install the required dependencies, follow these steps:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Code

### Contraction Cost Calculations & Tensor Sparsity
The script `contraction_cost_calculations.py` runs the Cotengra optimization and collects the intermediate tensor sparsities if specified. All command-line arguments are optional and have defaults if not included.

The command line arguments for this script are:
| Argument | Type | Default | Description |
|-----------|------|----------|--------------|
| `--file_name` | `str` | `"contraction_costs.csv"` | Name of the CSV file to save results. |
| `--num_runs` | `int` | `100` | Number of repeated runs for each tensor network configuration. Useful for averaging contraction costs or measuring variability. |
| `--methods` | `str` (list) | `["greedy", "kahypar"]` | List of contraction path optimization methods to run. Options: `greedy`, `kahypar`. |
| `--codes` | `str` (list) | `["concatenated", "rotated", "rotated_msp", "rotated_tanner", "hamming_msp", "hamming_tanner", "holo", "bb_msp", "bb_tanner"]` | Tensor network codes to test. Each corresponds to a different network topology. |
| `--max_time` | `int` | `None` | Maximum runtime per code (in seconds). |
| `--max_repeats` | `int` | `128` | Maximum number of trials Cotengra can perform per network before stopping. |
| `--sparsity_collection` | *(flag)* | `False` | When set, the script collects **tensor sparsity statistics** in addition to contraction cost data. |

To gather the same data as shown in the paper, run the following command:
```bash
python src/contraction_cost_calculations.py --file_name "contraction_costs.csv" --num_runs 100 --sparsity_collection
```

### WEP Contraction Costs (Scatter Plot)
This script runs the Cotengra optimization and the WEP calculation. This data is used in the paper to compare the Cotengra dense cost to the custom SST cost. This script has the same command-line argument options and defaults as above.

To gather the same data as shown in the paper, run the following command:
```bash
python src/wep_calculations.py --file_name "wep_calculations.csv" --num_runs 100 --minimize "flops" --methods "greedy"
```

### Optimal Costs
This script runs Cotengra's OptimalOptimizer for 4 small codes. Any larger or more complex codes were not feasible to compute optimally. The only command-line argument is the file_name.

```bash
python src/get_optimal_costs.py --file_name "optimal_costs.csv"
```

### Generate Plots
To create the visualizations, run:
```bash
python src/plotting_functions.py
```

There are five plotting functions in this script, each corresponding to a plot from the paper. The current paths used are the pregenerated data in `results/data`. Make sure to update the script if wanting to use other data.
