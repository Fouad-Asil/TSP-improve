#!/usr/bin/env python
"""
Script to summarize comparison results from multiple graph sizes.
Reads JSON files from a specified directory and prints a comparison table.
"""

import os
import json
import argparse
import glob
import pandas as pd

def summarize_results(results_dir):
    """
    Load results from JSON files, calculate differences, and print a summary table.
    
    Args:
        results_dir: Directory containing comparison_results_SIZE.json files.
    """
    
    all_results = []
    json_files = glob.glob(os.path.join(results_dir, "comparison_results_*.json"))
    
    if not json_files:
        print(f"Error: No comparison result JSON files found in {results_dir}")
        return
        
    print(f"Found {len(json_files)} result files.")
    
    for f_path in sorted(json_files):
        try:
            # Extract size from filename
            size = int(os.path.basename(f_path).split('_')[-1].split('.')[0])
            
            with open(f_path, 'r') as f:
                data = json.load(f)
                
                if 'baseline' not in data or 'overlap' not in data:
                    print(f"Warning: Skipping invalid file {f_path} - missing keys")
                    continue
                
                baseline = data['baseline']
                overlap = data['overlap']
                
                # Calculate difference (overlap - baseline)
                cost_diff = overlap['final_cost'] - baseline['final_cost']
                # Calculate percentage improvement difference (positive means overlap is better)
                # Handle potential division by zero if baseline cost is zero
                cost_diff_percent = (cost_diff / baseline['final_cost'] * 100) if baseline['final_cost'] != 0 else 0
                
                # Higher improvement % is better, so overlap - baseline
                improve_diff_percent = overlap['improvement_percent'] - baseline['improvement_percent']
                
                # Lower time is better, so baseline - overlap
                time_diff = baseline['time'] - overlap['time']
                
                all_results.append({
                    'Size': size,
                    'Baseline Cost': baseline['final_cost'],
                    'Overlap Cost': overlap['final_cost'],
                    'Cost Diff (%)': cost_diff_percent,
                    'Baseline Impr %': baseline['improvement_percent'],
                    'Overlap Impr %': overlap['improvement_percent'],
                    'Impr Diff (%pts)': improve_diff_percent,
                    'Baseline Time (s)': baseline['time'],
                    'Overlap Time (s)': overlap['time']
                })
                
        except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            print(f"Warning: Skipping file {f_path} due to error: {e}")
            
    if not all_results:
        print("Error: No valid results found to summarize.")
        return
        
    # Create DataFrame and sort by size
    df = pd.DataFrame(all_results)
    df = df.sort_values(by='Size').reset_index(drop=True)
    
    # Print the summary table
    print("\n" + "="*120)
    print("Multi-Size Comparison Summary")
    print("="*120)
    
    # Format floats for better readability
    pd.options.display.float_format = '{:.4f}'.format
    df_display = df.copy()
    df_display['Cost Diff (%)'] = df_display['Cost Diff (%)'].map('{:.2f}%'.format)
    df_display['Baseline Impr %'] = df_display['Baseline Impr %'].map('{:.2f}%'.format)
    df_display['Overlap Impr %'] = df_display['Overlap Impr %'].map('{:.2f}%'.format)
    df_display['Impr Diff (%pts)'] = df_display['Impr Diff (%pts)'].map('{:.2f} %pts'.format)

    print(df_display.to_string())
    print("="*120)
    print("Negative 'Cost Diff (%)' means Edge Overlap method produced shorter tours.")
    print("Positive 'Impr Diff (%pts)' means Edge Overlap method had higher improvement percentage.")
    print("="*120)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize multi-size comparison results.")
    parser.add_argument('--results_dir', type=str, default='./experiment_logs', 
                        help='Directory containing the comparison_results_SIZE.json files')
    
    args = parser.parse_args()
    
    summarize_results(args.results_dir) 