import mne
import numpy as np
import os
import pandas as pd

def analyze_single_epoch_file(file_path, lock_type):
    """Analyze a single epoch file and return timing information."""
    print(f"\nAnalyzing {os.path.basename(file_path)}")
    print("-" * 50)
    
    # Load epochs
    epochs = mne.read_epochs(file_path, preload=True)
    
    # Basic information
    print(f"Lock type: {lock_type}")
    print(f"Event IDs: {epochs.event_id}")
    print(f"Total epochs: {len(epochs)}")
    print(f"Time range: {epochs.tmin:.3f} to {epochs.tmax:.3f} seconds")
    
    results = []
    # Analyze each condition
    for condition_name, event_id in epochs.event_id.items():
        condition_epochs = epochs[condition_name]
        
        # Get data and calculate statistics
        data = condition_epochs.get_data()
        mean_signal = np.mean(data, axis=0)  # average across trials
        mean_signal = np.mean(mean_signal, axis=0)  # average across channels
        
        # Find the main peak after time zero
        zero_idx = np.where(epochs.times >= 0)[0][0]
        peak_idx = zero_idx + np.argmax(np.abs(mean_signal[zero_idx:]))
        peak_time = epochs.times[peak_idx]
        peak_amplitude = mean_signal[peak_idx]
        
        # Get event timing information
        event_times = condition_epochs.events[:, 0] / epochs.info['sfreq']
        
        result = {
            'lock_type': lock_type,
            'condition': condition_name,
            'n_trials': len(condition_epochs),
            'mean_event_time': np.mean(event_times),
            'std_event_time': np.std(event_times),
            'peak_time': peak_time,
            'peak_amplitude': peak_amplitude,
            'first_event_time': np.min(event_times),
            'last_event_time': np.max(event_times)
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Print condition comparison if we have multiple conditions
    if len(results_df) > 1:
        cond1, cond2 = results_df.iloc[0], results_df.iloc[1]
        print(f"\nComparison between {cond1['condition']} and {cond2['condition']}:")
        print(f"Trial counts: {cond1['n_trials']} vs {cond2['n_trials']}")
        print(f"Peak timing difference: {(cond1['peak_time'] - cond2['peak_time'])*1000:.2f} ms")
        print(f"Mean event time difference: {(cond1['mean_event_time'] - cond2['mean_event_time'])*1000:.2f} ms")
    
    return results_df

def analyze_participant(participant_dir):
    """Analyze all epoch files for a participant."""
    all_results = []
    
    # Define the analysis structure
    analysis_folders = {
        'abs_pe_epochs': ['R1_Locked', 'R2_Locked', 'R3_Locked', 'Outcome_Locked'],
        'pe_sign_epochs': ['R1_Locked', 'R2_Locked', 'R3_Locked', 'Outcome_Locked'],
        'vep_epochs': ['Stim_Locked', 'Outcome_All_Locked']
    }
    
    for folder, lock_types in analysis_folders.items():
        folder_path = os.path.join(participant_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder} not found")
            continue
            
        print(f"\nProcessing {folder}")
        print("=" * 50)
        
        for lock_type in lock_types:
            file_name = f"{lock_type}-prepro_epochs-epo.fif"
            file_path = os.path.join(folder_path, file_name)
            
            if os.path.exists(file_path):
                try:
                    results = analyze_single_epoch_file(file_path, lock_type)
                    all_results.append(results)
                except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
            else:
                print(f"Warning: File {file_name} not found")
    
    # Combine all results
    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        return all_results_df
    return None

def main():
    base_dir = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.28\Pilot Data Analysis\Enhanced Epoching"
    
    for participant in ['Elmira', 'Harry']:
        print(f"\nAnalyzing {participant}'s data")
        print("=" * 70)
        
        participant_dir = os.path.join(base_dir, participant)
        if not os.path.exists(participant_dir):
            print(f"Error: Directory not found for {participant}")
            continue
        
        results = analyze_participant(participant_dir)
        
        if results is not None:
            # Save results to CSV in the script's directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_file = os.path.join(script_dir, f'{participant}_epoch_analysis_results.csv')
            results.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
            
            # Print summary of timing differences
            print("\nSummary of condition differences:")
            for lock_type in results['lock_type'].unique():
                lock_data = results[results['lock_type'] == lock_type]
                if len(lock_data) > 1:
                    conditions = lock_data['condition'].values
                    times = lock_data['peak_time'].values
                    print(f"\n{lock_type}:")
                    print(f"Peak timing difference ({conditions[0]} vs {conditions[1]}): "
                          f"{(times[0] - times[1])*1000:.2f} ms")

if __name__ == "__main__":
    main()