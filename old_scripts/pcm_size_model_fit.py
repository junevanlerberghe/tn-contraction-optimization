import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Data
df = pd.read_csv('contraction_steps_with_open_legs.csv', sep = ';')

# Parse the '(r,c)' strings into two columns
df[['rows', 'cols']] = df['new pcm size'].str.extract(r'\((\d+),\s*(\d+)\)').astype(int)
costs_array = df['cost'].to_numpy()
rows = df['rows'].to_numpy()
cols = df['cols'].to_numpy()
x = (df["rows"] + df["cols"] + np.minimum(df["rows"], df["cols"])).to_numpy()

products = rows * cols

# Model: cost = a * (rows * cols)^b
def model(x, a, b):
    return a * (b ** x)

# Fit the model
params, covariance = curve_fit(model, x, costs_array)
a, b = params

# Generate fit line
x_fit = np.linspace(min(x), max(x), 100)
y_fit = model(x_fit, a, b)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(x, costs_array, label="Actual Data", color="blue")
plt.plot(x_fit, y_fit, label=f"Fit: cost = {a:.2f} * ({b:.2f}^rows)", color="red")
plt.xlabel("rows * cols")
plt.ylabel("Cost")
plt.title("Cost vs. Matrix Size (rows * cols)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
