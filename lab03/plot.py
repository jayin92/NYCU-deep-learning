import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# CSV data provided
csv_data = """Total Iterations,FID Score,Generation Time (s),Total Time (s)
1,36.181750011695385,50.90,69.57
2,32.55400842989246,75.62,96.18
3,32.638621627270055,83.53,103.76
4,31.601959356495,105.88,125.12
5,31.961801300456727,115.15,137.03
6,32.005403711508734,127.27,147.75
7,32.25398252578228,151.51,173.07
8,30.5605639528884,157.28,176.25
9,31.678650446092945,166.88,185.39
10,31.178892861825375,178.40,199.05
11,31.631070187601836,191.86,209.91
12,30.834626762710087,207.31,223.89
13,31.687284064675595,221.30,239.56
14,32.86186073784762,235.62,252.84
15,31.53866752937998,253.30,271.02
16,32.01877751054013,273.30,290.96
17,32.5255543632768,298.88,344.91
18,32.07504767368397,311.61,333.17
19,32.34560069247868,328.20,349.91
20,31.62758787250496,330.42,350.09
21,31.121364129634657,416.60,435.31
22,31.313489694420724,458.33,481.50
23,31.44780247561235,416.63,435.66
24,32.176256983626274,436.83,455.91
25,31.691255817547983,420.89,440.61
26,31.16656071483078,444.22,462.48
27,30.900759371400568,455.06,472.69
28,32.05722454203996,445.55,463.99"""

# Read the CSV data using StringIO to simulate a file
df = pd.read_csv(io.StringIO(csv_data))

# Create a figure and primary y-axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot FID Score on primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Total Iterations')
ax1.set_ylabel('FID Score', color=color)
ax1.plot(df['Total Iterations'], df['FID Score'], marker='o', linestyle='-', color=color, label='FID Score')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# Find the minimum FID score and its corresponding iteration
min_fid_idx = df['FID Score'].idxmin()
min_fid = df['FID Score'].min()
min_fid_iter = df.loc[min_fid_idx, 'Total Iterations']

# Highlight the minimum FID score point
ax1.scatter(min_fid_iter, min_fid, color='red', s=100, zorder=5)
ax1.annotate(f'Min FID: {min_fid:.2f}\nIter: {min_fid_iter}', 
             (min_fid_iter, min_fid),
             xytext=(-60, 15),
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='red'))

# Create a secondary y-axis for time measurements
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Time (s)', color=color)
ax2.plot(df['Total Iterations'], df['Generation Time (s)'], marker='s', linestyle='--', color=color, label='Generation Time')
ax2.tick_params(axis='y', labelcolor=color)

# Add a title and legend
plt.title('FID Score and Execution Time vs Total Iterations', fontsize=14)

# Create a combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# Adjust layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

# Save the figure
plt.savefig('fid_time_plot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print("Plot saved as 'fid_time_plot.png'")