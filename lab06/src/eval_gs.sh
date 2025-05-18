#!/bin/bash

# Create a results directory to store all experiment outputs
mkdir -p guidance_scale_results

# Define the guidance scales to test
guidance_scales=(0.0 1.5 3.5 5.5 7.5 9.5 11.5 13.5 15.5)

# Track accuracies for later visualization
echo "Scale,Test Accuracy,New Test Accuracy" > guidance_scale_results/accuracies.csv

# Loop through each guidance scale
for scale in "${guidance_scales[@]}"; do
  echo "===== Testing guidance scale $scale ====="
  
  # Create a directory for this scale's results
  output_dir="guidance_scale_results/scale_${scale}"
  mkdir -p "$output_dir"
  
  # Run the main.py with the current scale
  python src/main.py --test \
    --output_dir "$output_dir" \
    --beta_schedule cosine \
    --checkpoint output_new_adagn/checkpoints/final.pth \
    --guidance_scale "$scale" \
    --use_adagn \
    --use_classifier_guidance
  
  # Extract accuracies from the results.txt file (assuming it contains the accuracies)
  if [ -f "$output_dir/results.txt" ]; then
    test_accuracy=$(grep "test.json" "$output_dir/results.txt" | grep -oE "0\.[0-9]+")
    new_test_accuracy=$(grep "new_test.json" "$output_dir/results.txt" | grep -oE "0\.[0-9]+")
    echo "$scale,$test_accuracy,$new_test_accuracy" >> guidance_scale_results/accuracies.csv
  fi
  
  # Copy the grids to a common directory for easy comparison
  mkdir -p guidance_scale_results/grids
  if [ -f "$output_dir/grids/test_grid.png" ]; then
    cp "$output_dir/grids/test_grid.png" "guidance_scale_results/grids/test_grid_scale_${scale}.png"
  fi
  if [ -f "$output_dir/grids/new_test_grid.png" ]; then
    cp "$output_dir/grids/new_test_grid.png" "guidance_scale_results/grids/new_test_grid_scale_${scale}.png"
  fi
  
  echo "===== Completed guidance scale $scale ====="
  echo ""
done

# Generate a LaTeX table
echo "Generating LaTeX table..."
cat > guidance_scale_results/guidance_scale_table.tex << EOF
\\begin{table}[h]
\\centering
\\caption{Effect of Classifier Guidance Scale on Accuracy}
\\label{tab:guidance_scale}
\\begin{tabular}{ccc}
\\toprule
\\textbf{Guidance Scale} & \\textbf{Test Accuracy} & \\textbf{New Test Accuracy} \\\\
\\midrule
EOF

# Add data rows to the LaTeX table
tail -n +2 guidance_scale_results/accuracies.csv | while IFS=, read -r scale test_acc new_test_acc; do
  echo "$scale & $test_acc & $new_test_acc \\\\" >> guidance_scale_results/guidance_scale_table.tex
done

# Complete the LaTeX table
cat >> guidance_scale_results/guidance_scale_table.tex << EOF
\\bottomrule
\\end{tabular}
\\end{table}
EOF

echo "All experiments completed! Results are stored in guidance_scale_results/"
echo "Accuracy data is available in guidance_scale_results/accuracies.csv"
echo "LaTeX table saved to guidance_scale_results/guidance_scale_table.tex"