import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# Raw data with multiple runs per iteration
raw_data = """Total Iterations,Run,FID Score,Generation Time (s),Total Time (s)
1,1,34.56580945861293,37.170761657,53.795722181
1,2,35.36521577454127,33.915156583,48.451509670
1,3,35.80412597261636,40.828819306,56.388504401
1,4,36.11078713611096,39.079593607,54.680823231
1,5,35.21573997906759,39.474734814,53.833327278
2,1,32.945396090907536,48.113898824,58.024830617
2,2,33.09036848339582,32.894266535,43.581085555
2,3,33.38861794816739,34.739374258,45.549228383
2,4,32.93406897412763,32.030464967,44.158841766
2,5,33.765030904907064,32.541821309,42.404312389
3,1,32.907982996107904,45.326616267,56.622949042
3,2,33.28907704431904,41.623692799,50.742398162
3,3,32.81259501928744,50.045174223,60.643769339
3,4,32.53085260302473,55.558934142,66.067735994
3,5,33.02594226787372,44.794841048,54.605979736
4,1,32.01410432591581,48.303533288,58.820571386
4,2,31.915862828289818,46.919664679,57.804205155
4,3,31.882690330541635,60.797710736,72.734748806
4,4,32.748801834960204,54.797012359,66.587144561
4,5,32.97575557852491,55.229837935,67.393797788
5,1,31.787202743266107,80.895268565,94.860954543
5,2,32.39958554827956,143.250397190,161.944177730
5,3,32.61574937202758,191.330425031,209.447297868
5,4,31.41713289561858,70.155897580,84.372704396
5,5,32.25747617369282,67.755006919,80.106678555
6,1,31.920208472538008,73.495318838,84.749641655
6,2,32.10299244455277,82.206977132,94.022587492
6,3,31.70754615324944,74.785132058,86.186352279
6,4,32.3573929615396,70.398288652,81.919010921
6,5,31.266456944383293,77.152730593,88.682059730
7,1,32.00454719741953,94.307780732,106.957609627
7,2,31.327711895015852,71.872085376,83.951747249
7,3,31.51776518260715,77.742651769,89.327904693
7,4,31.669364174276666,69.972774917,81.032183086
7,5,31.390219146411255,79.979310727,93.993212881
8,1,31.703688399203628,116.093656566,132.497398339
8,2,32.18835333088657,96.629127157,112.217974916
8,3,31.62510395862907,93.809058069,107.491802944
8,4,32.63818753224214,113.920169186,128.692354076
8,5,31.223621440346562,84.221896172,96.390880914
9,1,31.510646722828767,98.804477497,110.146916232
9,2,32.42521175661409,89.924126213,101.437346322
9,3,31.995773960758726,114.517313102,137.144453514
9,4,30.934397926634773,110.882807404,133.642117957
9,5,32.24682654332747,132.317475149,145.985027891
10,1,31.051723559457287,138.482872585,153.474463315
10,2,32.06234477404195,113.143404327,126.722385577
10,3,32.240469849117545,240.924662278,254.630536642
10,4,31.585370348779833,123.452021810,138.835678339
10,5,31.813750032247555,145.145926875,158.817580236"""

# Read the raw data with multiple runs
df_raw = pd.read_csv(io.StringIO(raw_data))

# Group by Total Iterations and calculate statistics
df_stats = df_raw.groupby('Total Iterations').agg({
    'FID Score': ['mean', 'std'],
    'Generation Time (s)': ['mean', 'std'],
    'Total Time (s)': ['mean', 'std']
}).reset_index()

# Flatten the column names
df_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_stats.columns.values]

# Convert to a cleaner format for plotting
df = pd.DataFrame({
    'Total Iterations': df_stats['Total Iterations'],
    'FID Score': df_stats['FID Score_mean'],
    'FID Score Std': df_stats['FID Score_std'],
    'Generation Time (s)': df_stats['Generation Time (s)_mean'],
    'Generation Time Std': df_stats['Generation Time (s)_std'],
    'Total Time (s)': df_stats['Total Time (s)_mean'],
    'Total Time Std': df_stats['Total Time (s)_std']
})

# Create a figure and primary y-axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot FID Score on primary y-axis with error bars
color = 'tab:blue'
ax1.set_xlabel('Total Iterations')
ax1.set_ylabel('FID Score', color=color)
ax1.errorbar(df['Total Iterations'], df['FID Score'], 
             yerr=df['FID Score Std'],
             marker='o', linestyle='-', color=color, 
             label='FID Score', capsize=5)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# Find the minimum FID score and its corresponding iteration
min_fid_idx = df['FID Score'].idxmin()
min_fid = df['FID Score'].min()
min_fid_iter = df.loc[min_fid_idx, 'Total Iterations']
min_fid_std = df.loc[min_fid_idx, 'FID Score Std']

# Highlight the minimum FID score point
ax1.scatter(min_fid_iter, min_fid, color='red', s=100, zorder=5)
ax1.annotate(f'Min FID: {min_fid:.2f} ± {min_fid_std:.2f}\nIter: {min_fid_iter}', 
             (min_fid_iter, min_fid),
             xytext=(-60, 15),
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='red'))

# Create a secondary y-axis for time measurements
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Time (s)', color=color)
ax2.errorbar(df['Total Iterations'], df['Generation Time (s)'], 
             yerr=df['Generation Time Std'],
             marker='s', linestyle='--', color=color, 
             label='Generation Time', capsize=5)
ax2.tick_params(axis='y', labelcolor=color)

# Add a title and legend
plt.title('FID Score and Execution Time vs Total Iterations (Average of 5 Runs)', fontsize=14)

# Create a combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# Adjust layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)

# Set x-axis to display integers only
ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Save the figure
plt.savefig('fid_time_plot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print("Plot saved as 'fid_time_plot.png'")