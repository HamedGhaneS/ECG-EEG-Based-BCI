"""
Trial Timing Sequence Checker
----------------------------
Author: Hamed Ghane
Date: January 17, 2025

This script analyzes the timing sequence of each trial to verify:
- Trial start
- Choice made time
- Outcome presentation
- Three subsequent R-peaks
And checks for any timing inconsistencies.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def check_trial_sequence(timing_df, behavioral_df):
    """
    Check timing sequence for each trial and generate detailed log.
    Returns a DataFrame with all timing information and flags for irregularities.
    """
    trial_timings = []
    
    for idx, behav_trial in behavioral_df.iterrows():
        # Get corresponding timing data
        timing_trial = timing_df[
            (timing_df['block'] == behav_trial['block']) & 
            (timing_df['trial'] == behav_trial['trial'])
        ]
        
        # Skip if no matching timing data found
        if len(timing_trial) == 0:
            print(f"Warning: No timing data found for Block {behav_trial['block']}, Trial {behav_trial['trial']}")
            continue
            
        timing_trial = timing_trial.iloc[0]  # Get first matching row
        
        # Initialize timing sequence
        sequence = {
            'block': behav_trial['block'],
            'trial': behav_trial['trial'],
            'trial_start': behav_trial['trial_start_time'],
            'choice_made': behav_trial['choice_time'],
            'outcome_time': behav_trial['outcome_onset_time'],
            'r1_time': timing_trial['r1_time'] if pd.notna(timing_trial['r1_time']) else None,
            'r2_time': timing_trial['r2_time'] if pd.notna(timing_trial['r2_time']) else None,
            'r3_time': timing_trial['r3_time'] if pd.notna(timing_trial['r3_time']) else None,
            'timing_issues': []
        }
        
        # Only check timing if we have valid times
        if pd.notna(sequence['trial_start']) and pd.notna(sequence['choice_made']):
            # 1. Trial start should be before choice
            if sequence['choice_made'] <= sequence['trial_start']:
                sequence['timing_issues'].append("Choice before trial start")
                
            # 2. Choice should be before outcome
            if pd.notna(sequence['outcome_time']):
                if sequence['outcome_time'] <= sequence['choice_made']:
                    sequence['timing_issues'].append("Outcome before choice")
                
                # 3. Check R-peak sequence
                if sequence['r1_time'] is not None:
                    if sequence['r1_time'] <= sequence['outcome_time']:
                        sequence['timing_issues'].append("R1 before outcome")
                        
                    if sequence['r2_time'] is not None:
                        if sequence['r2_time'] <= sequence['r1_time']:
                            sequence['timing_issues'].append("R2 before R1")
                            
                        if sequence['r3_time'] is not None:
                            if sequence['r3_time'] <= sequence['r2_time']:
                                sequence['timing_issues'].append("R3 before R2")
        
        # Calculate intervals
        sequence['choice_to_outcome'] = (sequence['outcome_time'] - sequence['choice_made']) if pd.notna(sequence['choice_made']) and pd.notna(sequence['outcome_time']) else None
        sequence['outcome_to_r1'] = (sequence['r1_time'] - sequence['outcome_time']) if sequence['r1_time'] is not None and pd.notna(sequence['outcome_time']) else None
        sequence['r1_to_r2'] = (sequence['r2_time'] - sequence['r1_time']) if sequence['r1_time'] is not None and sequence['r2_time'] is not None else None
        sequence['r2_to_r3'] = (sequence['r3_time'] - sequence['r2_time']) if sequence['r2_time'] is not None and sequence['r3_time'] is not None else None
        
        trial_timings.append(sequence)
    
    return trial_timings

def save_timing_log(trial_timings, output_path):
    """Save timing analysis to a detailed log file."""
    with open(output_path, 'w') as f:
        f.write("Trial Timing Sequence Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        for trial in trial_timings:
            f.write(f"Block {trial['block']}, Trial {trial['trial']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Trial Start:      {trial['trial_start']:.3f}s\n")
            f.write(f"Choice Made:      {trial['choice_made']:.3f}s  (+{(trial['choice_made'] - trial['trial_start']):.3f}s)\n")
            f.write(f"Outcome:          {trial['outcome_time']:.3f}s  (+{(trial['outcome_time'] - trial['choice_made']):.3f}s)\n")
            
            if trial['r1_time'] is not None:
                f.write(f"R1:              {trial['r1_time']:.3f}s  (+{(trial['r1_time'] - trial['outcome_time']):.3f}s)\n")
            if trial['r2_time'] is not None:
                f.write(f"R2:              {trial['r2_time']:.3f}s  (+{(trial['r2_time'] - trial['r1_time']):.3f}s)\n")
            if trial['r3_time'] is not None:
                f.write(f"R3:              {trial['r3_time']:.3f}s  (+{(trial['r3_time'] - trial['r2_time']):.3f}s)\n")
            
            f.write("\nIntervals:\n")
            f.write(f"Choice to Outcome: {trial['choice_to_outcome']:.3f}s\n")
            if trial['outcome_to_r1'] is not None:
                f.write(f"Outcome to R1:     {trial['outcome_to_r1']:.3f}s\n")
            if trial['r1_to_r2'] is not None:
                f.write(f"R1 to R2:          {trial['r1_to_r2']:.3f}s\n")
            if trial['r2_to_r3'] is not None:
                f.write(f"R2 to R3:          {trial['r2_to_r3']:.3f}s\n")
            
            if trial['timing_issues']:
                f.write("\nTIMING ISSUES DETECTED:\n")
                for issue in trial['timing_issues']:
                    f.write(f"! {issue}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")

def main():
    # Load data
    base_path = Path(r"H:\Post\6th_Phase (ECG-EEG Baced BCI)")
    timing_file = base_path / "2025.01.17/Pilot Data Analysis/timing_analysis/trial_timing_analysis_20250117_150100.csv"
    behavioral_file = base_path / "2025.01.14/Pilot/Harry/Harry-ses001-run1-20250114-130445.csv"
    
    # Load data
    timing_df = pd.read_csv(timing_file)
    behavioral_df = pd.read_csv(behavioral_file)
    
    # Check timing sequences
    trial_timings = check_trial_sequence(timing_df, behavioral_df)
    
    # Save detailed log
    output_path = base_path / "2025.01.17/Pilot Data Analysis/timing_analysis/trial_timing_log.txt"
    save_timing_log(trial_timings, output_path)
    
    # Print summary
    n_trials = len(trial_timings)
    n_issues = len([t for t in trial_timings if t['timing_issues']])
    print(f"\nTiming Analysis Summary:")
    print(f"Total trials analyzed: {n_trials}")
    print(f"Trials with timing issues: {n_issues}")
    if n_issues > 0:
        print("\nPlease check the detailed log file for specific timing issues.")
    print(f"\nDetailed log saved to: {output_path}")

if __name__ == "__main__":
    main()