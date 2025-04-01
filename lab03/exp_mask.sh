#!/bin/bash

# Create output directory for results
mkdir -p fid_results

# Output files
output_file="fid_results/fid_scores.txt"
csv_output="fid_results/fid_scores.csv"
detailed_csv="fid_results/detailed_results.csv"

# Number of runs per mask function setting
runs=5

# Fix total iterations
total_iter=7

# Mask functions to test
mask_functions=("linear" "cosine" "square")

# Clear previous results
echo "Mask Function,FID Score,FID StdDev,Generation Time (s),Gen Time StdDev,Total Time (s),Total Time StdDev" > $csv_output
echo "Mask Function,Run,FID Score,Generation Time (s),Total Time (s)" > $detailed_csv
echo "FID scores and timing for different mask scheduling functions (average of $runs runs, total iterations: $total_iter):" > $output_file

# Function to extract FID score from output
extract_fid() {
    local output_text=$1
    echo "$output_text" | grep -o "FID: *[0-9.]*" | awk '{print $2}'
}

# Function to calculate average
calculate_average() {
    local sum=0
    local count=0
    for value in "$@"; do
        sum=$(echo "$sum + $value" | bc -l)
        count=$((count + 1))
    done
    if [ $count -eq 0 ]; then
        echo "0"
    else
        echo "scale=6; $sum / $count" | bc -l
    fi
}

# Function to calculate standard deviation
calculate_stddev() {
    local avg=$1
    shift
    local sum_sq_diff=0
    local count=0
    for value in "$@"; do
        diff=$(echo "$value - $avg" | bc -l)
        sq_diff=$(echo "$diff * $diff" | bc -l)
        sum_sq_diff=$(echo "$sum_sq_diff + $sq_diff" | bc -l)
        count=$((count + 1))
    done
    if [ $count -le 1 ]; then
        echo "0"
    else
        echo "scale=6; sqrt($sum_sq_diff / ($count - 1))" | bc -l
    fi
}

# Run for each mask function
for mask_func in "${mask_functions[@]}"; do
    echo "Running with --mask-func=$mask_func ($runs times, total_iter=$total_iter)"
    
    # Arrays to store results
    fid_scores=()
    gen_times=()
    total_times=()
    
    # Run multiple times for each mask function setting
    for run in $(seq 1 $runs); do
        echo "  Run $run/$runs..."
        
        # Run the inpainting script with current mask function and fixed total iterations
        start_gen_time=$(date +%s.%N)
        output=$(python inpainting.py --total-iter=$total_iter --mask-func=$mask_func 2>&1)
        end_gen_time=$(date +%s.%N)
        gen_time=$(echo "$end_gen_time - $start_gen_time" | bc)
        
        # Change to FID directory and run FID calculation
        cd faster-pytorch-fid/
        start_fid_time=$(date +%s.%N)
        fid_output=$(python fid_score_gpu.py --predicted-path ../test_results/ 2>&1)
        end_fid_time=$(date +%s.%N)
        cd ..
        
        # Calculate total time
        total_time=$(echo "$end_fid_time - $start_gen_time" | bc)
        
        # Extract FID score
        fid_score=$(extract_fid "$fid_output")
        
        # Store results in arrays
        fid_scores+=($fid_score)
        gen_times+=($gen_time)
        total_times+=($total_time)
        
        # Log detailed results
        echo "$mask_func,$run,$fid_score,$gen_time,$total_time" >> $detailed_csv
        
        # Optional: save the generated images for the last run of each mask function
        if [ $run -eq $runs ] && [ -d "test_results" ]; then
            mkdir -p "fid_results/${mask_func}_samples"
            cp test_results/* "fid_results/${mask_func}_samples/"
        fi
    done
    
    # Calculate averages
    avg_fid=$(calculate_average "${fid_scores[@]}")
    avg_gen_time=$(calculate_average "${gen_times[@]}")
    avg_total_time=$(calculate_average "${total_times[@]}")
    
    # Calculate standard deviations
    stddev_fid=$(calculate_stddev "$avg_fid" "${fid_scores[@]}")
    stddev_gen_time=$(calculate_stddev "$avg_gen_time" "${gen_times[@]}")
    stddev_total_time=$(calculate_stddev "$avg_total_time" "${total_times[@]}")
    
    # Format results to 2 decimal places
    avg_fid_fmt=$(printf "%.2f" $avg_fid)
    avg_gen_time_fmt=$(printf "%.2f" $avg_gen_time)
    avg_total_time_fmt=$(printf "%.2f" $avg_total_time)
    stddev_fid_fmt=$(printf "%.2f" $stddev_fid)
    stddev_gen_time_fmt=$(printf "%.2f" $stddev_gen_time)
    stddev_total_time_fmt=$(printf "%.2f" $stddev_total_time)
    
    # Log the summarized results
    echo "Mask Function: $mask_func, FID: $avg_fid_fmt ± $stddev_fid_fmt, Generation Time: ${avg_gen_time_fmt}s ± ${stddev_gen_time_fmt}s, Total Time: ${avg_total_time_fmt}s ± ${stddev_total_time_fmt}s" >> $output_file
    echo "$mask_func,$avg_fid_fmt,$stddev_fid_fmt,$avg_gen_time_fmt,$stddev_gen_time_fmt,$avg_total_time_fmt,$stddev_total_time_fmt" >> $csv_output
    
    # Also print to console
    echo "Mask Function: $mask_func, FID: $avg_fid_fmt ± $stddev_fid_fmt, Generation Time: ${avg_gen_time_fmt}s ± ${stddev_gen_time_fmt}s, Total Time: ${avg_total_time_fmt}s ± ${stddev_total_time_fmt}s"
done

echo "All tests completed. Results saved to $output_file and $csv_output"

# Generate bar chart for mask functions if gnuplot is available
if command -v gnuplot > /dev/null; then
    echo "Generating plots with gnuplot..."
    
    # FID Score bar chart with error bars
    gnuplot <<EOF
    set terminal png size 800,600
    set output "fid_results/mask_func_fid_plot.png"
    set title "FID Score by Mask Function (Average of $runs runs, Total Iterations: $total_iter)"
    set xlabel "Mask Function"
    set ylabel "FID Score"
    set grid y
    set style data histogram
    set style histogram errorbars gap 2 lw 2
    set style fill solid 0.5
    set boxwidth 0.8
    set xtics rotate by -45
    set key off
    # Use column 2 for y values and column 3 for yerror
    plot "$csv_output" using 2:3:xtic(1) title "FID Score" lt rgb "#0060ad"
EOF

    # Generation Time bar chart with error bars
    gnuplot <<EOF
    set terminal png size 800,600
    set output "fid_results/mask_func_time_plot.png"
    set title "Generation Time by Mask Function (Average of $runs runs, Total Iterations: $total_iter)"
    set xlabel "Mask Function"
    set ylabel "Time (seconds)"
    set grid y
    set style data histogram
    set style histogram errorbars gap 2 lw 2
    set style fill solid 0.5
    set boxwidth 0.8
    set xtics rotate by -45
    set key off
    # Use column 4 for y values and column 5 for yerror
    plot "$csv_output" using 4:5:xtic(1) title "Generation Time" lt rgb "#dd181f"
EOF

    # Combined bar chart
    gnuplot <<EOF
    set terminal png size 1000,800
    set output "fid_results/mask_func_combined_plot.png"
    set multiplot layout 2,1 title "Mask Function Comparison (Total Iterations: $total_iter)"
    
    # FID Score subplot
    set title "FID Score by Mask Function"
    set xlabel ""
    set ylabel "FID Score"
    set grid y
    set style data histogram
    set style histogram errorbars gap 2 lw 2
    set style fill solid 0.5
    set boxwidth 0.8
    set xtics rotate by 0
    set key off
    plot "$csv_output" using 2:3:xtic(1) title "FID Score" lt rgb "#0060ad"
    
    # Generation Time subplot
    set title "Generation Time by Mask Function"
    set xlabel "Mask Function"
    set ylabel "Time (seconds)"
    set grid y
    plot "$csv_output" using 4:5:xtic(1) title "Generation Time" lt rgb "#dd181f"
    
    unset multiplot
EOF

    echo "Plots saved to fid_results directory"
else
    echo "gnuplot not found. Skipping plot generation."
fi