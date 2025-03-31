#!/bin/bash

# Create output directory for results
mkdir -p fid_results

# Output file for collected FID scores
output_file="fid_results/fid_scores.txt"
csv_output="fid_results/fid_scores.csv"

# Clear previous results
echo "Total Iterations,FID Score,Generation Time (s),Total Time (s)" > $csv_output
echo "FID scores and timing for different --total-iter values:" > $output_file

# Function to extract FID score from output
extract_fid() {
    local output_text=$1
    # Extract the FID score using grep and awk
    echo "$output_text" | grep -o "FID: *[0-9.]*" | awk '{print $2}'
}

# Run for iterations 1 to 30
for iter in $(seq 1 30); do
    echo "Running with --total-iter=$iter"
    
    # Run the inpainting script with current iteration value and measure time
    start_gen_time=$(date +%s.%N)
    output=$(python inpainting.py --total-iter=$iter 2>&1)
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
    
    # Format times to 2 decimal places
    gen_time=$(printf "%.2f" $gen_time)
    total_time=$(printf "%.2f" $total_time)
    
    # Extract FID score
    fid_score=$(extract_fid "$fid_output")
    
    # Log the results
    echo "Total Iterations: $iter, FID: $fid_score, Generation Time: ${gen_time}s, Total Time: ${total_time}s" >> $output_file
    echo "$iter,$fid_score,$gen_time,$total_time" >> $csv_output
    
    # Also print to console
    echo "Total Iterations: $iter, FID: $fid_score, Generation Time: ${gen_time}s, Total Time: ${total_time}s"
    
    # Optional: save the generated images for each iteration in separate folders
    if [ -d "test_results" ]; then
        mkdir -p "fid_results/iter_${iter}_samples"
        cp test_results/* "fid_results/iter_${iter}_samples/"
    fi
done

echo "All tests completed. Results saved to $output_file and $csv_output"

# Generate plots if gnuplot is available
if command -v gnuplot > /dev/null; then
    echo "Generating plots with gnuplot..."
    
    # FID Score plot
    gnuplot <<EOF
    set terminal png size 800,600
    set output "fid_results/fid_plot.png"
    set title "FID Score vs. Total Iterations"
    set xlabel "Total Iterations"
    set ylabel "FID Score"
    set grid
    set key outside
    plot "$csv_output" using 1:2 with linespoints lw 2 pt 7 ps 1 title "FID Score"
EOF

    # Generation Time plot
    gnuplot <<EOF
    set terminal png size 800,600
    set output "fid_results/generation_time_plot.png"
    set title "Generation Time vs. Total Iterations"
    set xlabel "Total Iterations"
    set ylabel "Time (seconds)"
    set grid
    set key outside
    plot "$csv_output" using 1:3 with linespoints lw 2 pt 7 ps 1 title "Generation Time"
EOF

    # Combined plot with two y-axes
    gnuplot <<EOF
    set terminal png size 1000,600
    set output "fid_results/combined_plot.png"
    set title "FID Score and Generation Time vs. Total Iterations"
    set xlabel "Total Iterations"
    set ylabel "FID Score"
    set y2label "Time (seconds)"
    set ytics nomirror
    set y2tics
    set grid
    set key outside
    
    # Set different colors for the two plots
    set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 ps 1   # blue
    set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 ps 1   # red
    
    plot "$csv_output" using 1:2 with linespoints ls 1 title "FID Score" axis x1y1, \
         "$csv_output" using 1:3 with linespoints ls 2 title "Generation Time" axis x1y2
EOF

    echo "Plots saved to fid_results directory"
else
    echo "gnuplot not found. Skipping plot generation."
fi