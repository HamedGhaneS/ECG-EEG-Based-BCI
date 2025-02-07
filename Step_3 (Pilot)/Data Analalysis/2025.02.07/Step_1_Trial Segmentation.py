import pandas as pd
import numpy as np
from pathlib import Path
import os

def analyze_trial_timing(behavioral_data, timing_pairs_df):
    """
    Analyze timing sequence for all trials.
    """
    timing_info = []
    
    for idx, trial in behavioral_data.iterrows():
        # Get basic trial info
        block = trial['block']
        trial_num = trial['trial']
        feedback = trial['feedback']
        
        # Skip trials with no response
        if pd.isna(trial['choice_time']):
            print(f"\nBlock {block}, Trial {trial_num}: No response (timeout)")
            continue
            
        # Get key timing points
        fixation_end = trial['post_choice_fixation_time'] + 1.0  # 1s fixed duration
        valid_rpeak = trial['valid_rpeak_time']
        feedback_onset = trial['outcome_onset_time']
        
        # Get cardiac timing info
        cardiac_phase = trial['cardiac_phase']
        timing_point = trial['block_condition']
        percentage_rr = trial['percentage_rr']
        
        # Find the next 3 R-peaks after feedback onset
        post_feedback_peaks = timing_pairs_df[
            timing_pairs_df['pc2_lsl_time'] > feedback_onset
        ]['pc2_lsl_time'].values[:3]
        
        # Create trial timing dictionary
        trial_timing = {
            'block': block,
            'trial': trial_num,
            'feedback': feedback,
            'fixation_end': fixation_end,
            'valid_rpeak': valid_rpeak,
            'feedback_onset': feedback_onset,
            'cardiac_phase': cardiac_phase,
            'timing_point': timing_point,
            'percentage_rr': percentage_rr,
            'n_peaks_found': len(post_feedback_peaks)
        }
        
        # Add the three R-peaks if found
        for i, peak in enumerate(post_feedback_peaks, 1):
            trial_timing[f'r{i}_time'] = peak
            if i > 1:
                trial_timing[f'r{i-1}_to_r{i}_interval'] = peak - post_feedback_peaks[i-2]
        
        timing_info.append(trial_timing)
        
        # Print detailed info for this trial
        print(f"\nBlock {block}, Trial {trial_num}:")
        print(f"Condition: {cardiac_phase}-{timing_point} ({percentage_rr*100:.1f}% R-R)")
        print(f"Feedback: {feedback}")
        print("\nTiming Sequence:")
        print(f"1. Fixation End:     {fixation_end:.3f}")
        print(f"2. Valid R-peak:     {valid_rpeak:.3f} (+" + 
              f"{(valid_rpeak - fixation_end)*1000:.1f}ms after fixation)")
        print(f"3. Feedback Onset:   {feedback_onset:.3f} (+" +
              f"{(feedback_onset - valid_rpeak)*1000:.1f}ms after valid R-peak)")
        
        print("\nHEP R-peaks:")
        for i, peak in enumerate(post_feedback_peaks, 1):
            print(f"R{i}: {peak:.3f} (+" +
                  f"{(peak - feedback_onset)*1000:.1f}ms after feedback)")
            if i > 1:
                interval = (peak - post_feedback_peaks[i-2]) * 1000
                print(f"   R{i-1}-R{i} interval: {interval:.1f}ms")
    
    # Convert to DataFrame and return it
    return pd.DataFrame(timing_info)

def main():
    # Setup paths for input files
    base_path = Path(r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.02.07\3rd Pilot Data Analysis")
    
    # Input files (in the data directory)
    behavioral_file = base_path / "Raw Data/1-Behavioral-Task Data/Elmira-ses001-run1-20250131-185646.csv"
    timing_pairs_file = base_path / "Raw Data/1-Behavioral-Task Data/Elmira-ses001-run1-20250131-185646_timing_pairs.csv"
    
    # Create output directory in current working directory
    output_dir = base_path / "Timing_analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("Loading data files...")
    print(f"Behavioral data: {behavioral_file}")
    print(f"Timing pairs: {timing_pairs_file}")
    
    try:
        behavioral_data = pd.read_csv(behavioral_file)
        print(f"Loaded behavioral data: {len(behavioral_data)} trials")
        
        timing_pairs = pd.read_csv(timing_pairs_file)
        print(f"Loaded timing pairs: {len(timing_pairs)} R-peaks")
        
        # Analyze timing for all trials
        timing_df = analyze_trial_timing(behavioral_data, timing_pairs)
        
        if timing_df is not None and len(timing_df) > 0:
            # Save results in current directory
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"trial_timing_analysis_{timestamp}.csv"
            timing_df.to_csv(output_file, index=False)
            print(f"\nTiming analysis saved to: {output_file}")
            print(f"Current working directory: {os.getcwd()}")
            
            # Print summary statistics
            print("\nSummary Statistics:")
            print("-" * 50)
            
            # R-peak detection success
            total_trials = len(timing_df)
            trials_with_3_peaks = len(timing_df[timing_df['n_peaks_found'] == 3])
            print(f"Total trials analyzed: {total_trials}")
            print(f"Trials with all 3 R-peaks: {trials_with_3_peaks} " +
                  f"({trials_with_3_peaks/total_trials*100:.1f}%)")
            
            # Average intervals
            for i in range(1, 3):
                intervals = timing_df[f'r{i}_to_r{i+1}_interval'].dropna()
                if len(intervals) > 0:
                    print(f"\nR{i}-R{i+1} intervals:")
                    print(f"Mean: {intervals.mean()*1000:.1f}ms")
                    print(f"Std:  {intervals.std()*1000:.1f}ms")
                    print(f"Min:  {intervals.min()*1000:.1f}ms")
                    print(f"Max:  {intervals.max()*1000:.1f}ms")
        else:
            print("No timing data was generated!")
            
    except FileNotFoundError as e:
        print(f"Error: Could not find file: {e}")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        print("Full error trace:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()