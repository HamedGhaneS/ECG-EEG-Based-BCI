import numpy as np
import pandas as pd
from pathlib import Path
import json
import openpyxl

def check_trial_sequence(timing_df, behavioral_df, rl_data):
    """
    Check timing sequence for each trial and generate detailed log.
    Returns a list of dictionaries with all timing information and learning metrics.
    """
    trial_timings = []
    removed_trials = []

    for idx, behav_trial in behavioral_df.iterrows():
        # Skip if trial is invalid based on RL data or has "timed out" feedback
        if (not rl_data['trial_classifications']['valid_trials'][idx]) or \
           (pd.isna(behav_trial['feedback']) or behav_trial['feedback'].lower() == 'timed out'):
            removed_trials.append({
                'index': idx,
                'block': behav_trial.get('block', None),
                'trial': behav_trial.get('trial', None),
                'reason': 'Invalid trial' if not rl_data['trial_classifications']['valid_trials'][idx] else 'Timed out'
            })
            continue

        # Get corresponding timing data
        timing_trial = timing_df[
            (timing_df['block'] == behav_trial['block']) & 
            (timing_df['trial'] == behav_trial['trial'])
        ]

        # Skip if no matching timing data found
        if len(timing_trial) == 0:
            removed_trials.append({
                'index': idx,
                'block': behav_trial.get('block', None),
                'trial': behav_trial.get('trial', None),
                'reason': 'No timing data found'
            })
            continue

        timing_trial = timing_trial.iloc[0]  # Get first matching row

        # Determine trial correctness based on feedback
        trial_correct = 'correct' if behav_trial['feedback'] == 'win' else 'incorrect'

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

            # Trial performance
            'correct': trial_correct,
            'choice': behav_trial['choice'],
            'feedback': behav_trial['feedback'],
            'cardiac_phase': behav_trial['cardiac_phase'],
            'block_condition': behav_trial['block_condition'],

            # Add RL metrics
            'pe_sign': 'positive' if rl_data['trial_classifications']['positive_pe'][idx] else 'negative',
            'abs_pe_level': 'high' if rl_data['trial_classifications']['high_abs_pe'][idx] else 'low',
            'pe_value': rl_data['prediction_errors'][idx],
            'abs_pe_value': rl_data['abs_prediction_errors'][idx],

            'timing_issues': []
        }

        # Check timing sequence if we have valid times
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

    return trial_timings, removed_trials

def save_timing_log(trial_timings, removed_trials, output_path):
    """Save timing analysis to a detailed log file."""
    with open(output_path, 'w') as f:
        f.write("Trial Timing Sequence Analysis\n")
        f.write("=" * 80 + "\n\n")

        # Log removed trials
        f.write("Removed Trials:\n")
        f.write("-" * 80 + "\n")
        for trial in removed_trials:
            f.write(f"Index: {trial['index']}, Block: {trial['block']}, Trial: {trial['trial']}, Reason: {trial['reason']}\n")
        f.write("\n" + "=" * 80 + "\n\n")

        # Log valid trials
        for trial in trial_timings:
            f.write(f"Block {trial['block']}, Trial {trial['trial']}\n")
            f.write("-" * 50 + "\n")

            # Trial performance
            f.write(f"Correct:          {'Yes' if trial['correct'] == 'correct' else 'No'}\n")
            f.write(f"Choice:           {trial['choice']}\n")
            f.write(f"Feedback:         {trial['feedback']}\n")
            f.write(f"Cardiac Phase:    {trial['cardiac_phase']}\n")
            f.write(f"Block Condition:  {trial['block_condition']}\n\n")

            # Learning metrics
            f.write(f"PE Sign:          {trial['pe_sign']}\n")
            f.write(f"Absolute PE:      {trial['abs_pe_level']} ({trial['abs_pe_value']:.3f})\n")
            f.write(f"PE Value:         {trial['pe_value']:.3f}\n\n")

            # Timing information
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

def save_timing_excel(trial_timings, output_path):
    """
    Save timing analysis to an Excel file in a format suitable for EEG segmentation.
    Creates a structured DataFrame with all trial information.
    """
    # Convert trial timings list to DataFrame
    trials_data = []
    for trial in trial_timings:
        trial_data = {
            'block': trial['block'],
            'trial': trial['trial'],

            # Trial performance
            'correct': trial['correct'],
            'choice': trial['choice'],
            'feedback': trial['feedback'],
            'cardiac_phase': trial['cardiac_phase'],
            'block_condition': trial['block_condition'],

            # Timing information
            'trial_start_time': trial['trial_start'],
            'choice_time': trial['choice_made'],
            'outcome_time': trial['outcome_time'],
            'r1_time': trial['r1_time'],
            'r2_time': trial['r2_time'],
            'r3_time': trial['r3_time'],

            # Intervals
            'choice_to_outcome': trial['choice_to_outcome'],
            'outcome_to_r1': trial['outcome_to_r1'],
            'r1_to_r2': trial['r1_to_r2'],
            'r2_to_r3': trial['r2_to_r3'],

            # Learning metrics
            'pe_sign': trial['pe_sign'],
            'abs_pe_level': trial['abs_pe_level'],
            'pe_value': trial['pe_value'],
            'abs_pe_value': trial['abs_pe_value'],

            # Timing issues
            'timing_issues': '; '.join(trial['timing_issues']) if trial['timing_issues'] else 'none'
        }
        trials_data.append(trial_data)

    # Convert to DataFrame
    df = pd.DataFrame(trials_data)

    # Add some useful computed columns for EEG segmentation
    df['trial_duration'] = df['outcome_time'] - df['trial_start_time']
    df['choice_rt'] = df['choice_time'] - df['trial_start_time']

    # Sort by block and trial
    df = df.sort_values(['block', 'trial'])

    # Save to Excel with formatting
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Trial_Data', index=False)

def main():
    # Load data
    base_path = Path(r"H:\Post\6th_Phase (ECG-EEG Baced BCI)")
    timing_file = base_path / "2025.01.17/Pilot Data Analysis/timing_analysis/trial_timing_analysis_20250117_150100.csv"
    behavioral_file = base_path / "2025.01.14/Pilot/Harry/Harry-ses001-run1-20250114-130445.csv"
    rl_model_file = base_path / "2025.01.17/Pilot Data Analysis/rl_model_results/Harry_rl_model_results_130445.json"

    # Load data
    timing_df = pd.read_csv(timing_file)
    behavioral_df = pd.read_csv(behavioral_file)

    # Load RL model data
    with open(rl_model_file, 'r') as f:
        rl_data = json.load(f)

    # Check timing sequences
    trial_timings, removed_trials = check_trial_sequence(timing_df, behavioral_df, rl_data)

    # Save text log
    txt_output = base_path / "2025.01.20/Pilot Data Analysis/timing_analysis/trial_timing_log_with_removed.txt"
    save_timing_log(trial_timings, removed_trials, txt_output)

    # Save Excel file
    excel_output = base_path / "2025.01.20/Pilot Data Analysis/timing_analysis/trial_timing_data_cleaned.xlsx"
    save_timing_excel(trial_timings, excel_output)

    print(f"Timing analysis completed. Log saved to {txt_output}. Cleaned data saved to {excel_output}.")

if __name__ == "__main__":
    main()
