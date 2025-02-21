import mne
import pandas as pd
import numpy as np
from datetime import datetime

def get_cardiac_condition(row):
    """Combine cardiac_phase and block_condition into a single condition string"""
    return f"{row['cardiac_phase']}_{row['block_condition']}"

def validate_epochs_against_trials(epochs_path, timing_file_path, output_dir):
    """
    Validate epoch information against original trial data and generate a detailed report.
    Now uses combined cardiac_phase and block_condition for comparison.
    """
    # Load data
    print("Loading data...")
    epochs = mne.read_epochs(epochs_path)
    timing_data = pd.read_excel(timing_file_path)
    
    # Create validation report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f'epoch_validation_report_{timestamp}.txt'
    report_path = f"{output_dir}/{report_filename}"

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE EPOCH VALIDATION REPORT\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        # Basic count comparison
        f.write("1. COUNT COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of trials in timing file: {len(timing_data)}\n")
        f.write(f"Number of epochs: {len(epochs)}\n")
        f.write(f"Difference: {len(timing_data) - len(epochs)}\n\n")

        # Metadata comparison
        f.write("2. METADATA COMPARISON\n")
        f.write("-"*80 + "\n")
        
        # Compare cardiac phases using combined condition
        timing_data['cardiac_condition'] = timing_data.apply(get_cardiac_condition, axis=1)
        timing_conditions = timing_data['cardiac_condition'].unique()
        
        f.write("\nCardiac Conditions:\n")
        f.write("In timing data:\n")
        for cond in sorted(timing_conditions):
            f.write(f"  - {cond}\n")
        
        f.write("\nIn epochs:\n")
        for cond in sorted(epochs.event_id.keys()):
            f.write(f"  - {cond}\n")

        # Compare PE information
        if epochs.metadata is not None:
            f.write("\nPrediction Error Information:\n")
            f.write("Timing Data PE Stats:\n")
            f.write(f"Unique PE signs: {timing_data['pe_sign'].unique()}\n")
            f.write(f"AbsPE range: {timing_data['abs_pe_value'].min():.3f} to {timing_data['abs_pe_value'].max():.3f}\n")
            
            f.write("\nEpochs Metadata PE Stats:\n")
            f.write(f"Unique SignedPE: {epochs.metadata['SignedPE'].unique()}\n")
            f.write(f"AbsPE range: {epochs.metadata['AbsPE'].min():.3f} to {epochs.metadata['AbsPE'].max():.3f}\n")

        # Detailed trial-by-trial comparison
        f.write("\n3. COMPREHENSIVE TRIAL-BY-TRIAL COMPARISON\n")
        f.write("-"*80 + "\n")
        
        # Sort both datasets by r1_time/event time for comparison
        timing_sorted = timing_data.sort_values('r1_time').reset_index(drop=True)
        events_df = pd.DataFrame(epochs.events, columns=['sample', 'unused', 'event_id'])
        events_df['time'] = events_df['sample'] / epochs.info['sfreq']
        
        # Track inconsistencies
        inconsistencies = []
        missing_trials = []
        
        # Compare all trials
        max_trials = max(len(timing_sorted), len(events_df))
        for i in range(max_trials):
            inconsistent = False
            message = f"\nTrial {i+1}:\n"
            
            if i >= len(timing_sorted):
                message += "WARNING: Extra epoch found with no corresponding trial data\n"
                inconsistencies.append(i)
                continue
                
            if i >= len(events_df):
                message += "WARNING: Trial data has no corresponding epoch\n"
                missing_trials.append(i)
                continue
            
            # Get trial data
            trial = timing_sorted.iloc[i]
            event_id = events_df.iloc[i]['event_id']
            event_type = [k for k, v in epochs.event_id.items() if v == event_id][0]
            
            # Compare values
            timing_data_info = {
                'r1_time': trial['r1_time'],
                'cardiac_condition': trial['cardiac_condition'],
                'pe_sign': trial['pe_sign'],
                'abs_pe': trial['abs_pe_value']
            }
            
            epoch_data_info = {
                'time': events_df.iloc[i]['time'],
                'cardiac_condition': event_type,
                'SignedPE': epochs.metadata.iloc[i]['SignedPE'],
                'AbsPE': epochs.metadata.iloc[i]['AbsPE']
            }
            
            # Check for inconsistencies
            if not np.isclose(timing_data_info['abs_pe'], epoch_data_info['AbsPE'], atol=1e-5):
                inconsistent = True
            if timing_data_info['cardiac_condition'].lower() != epoch_data_info['cardiac_condition'].lower():
                inconsistent = True
            # Check PE signs, explicitly flagging zero PE cases with different labels
            if timing_data_info['abs_pe'] == 0 and timing_data_info['pe_sign'] != epoch_data_info['SignedPE']:
                inconsistent = True
                message += "*** Zero PE Sign Mismatch ***\n"
            elif timing_data_info['abs_pe'] > 0 and str(timing_data_info['pe_sign']).lower() != str(epoch_data_info['SignedPE']).lower():
                inconsistent = True
            
            if inconsistent:
                inconsistencies.append(i)
                message += "*** INCONSISTENCY DETECTED ***\n"
                message += "Timing data:\n"
                message += f"  r1_time: {timing_data_info['r1_time']:.3f}\n"
                message += f"  cardiac_condition: {timing_data_info['cardiac_condition']}\n"
                message += f"  pe_sign: {timing_data_info['pe_sign']}\n"
                message += f"  abs_pe: {timing_data_info['abs_pe']:.3f}\n"
                
                message += "Epoch data:\n"
                message += f"  time: {epoch_data_info['time']:.3f}\n"
                message += f"  cardiac_condition: {epoch_data_info['cardiac_condition']}\n"
                message += f"  SignedPE: {epoch_data_info['SignedPE']}\n"
                message += f"  AbsPE: {epoch_data_info['AbsPE']:.3f}\n"
                
                f.write(message)
        
        # Summary of inconsistencies
        f.write("\n4. INCONSISTENCY SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total trials checked: {max_trials}\n")
        
        # Count zero PE mismatches
        zero_pe_mismatches = sum(1 for i in range(len(timing_sorted)) 
                                if timing_sorted.iloc[i]['abs_pe_value'] == 0 
                                and timing_sorted.iloc[i]['pe_sign'] != epochs.metadata.iloc[i]['SignedPE'])
        
        f.write(f"Number of zero PE sign mismatches (negative vs Neutral): {zero_pe_mismatches}\n")
        f.write(f"Number of other inconsistent trials: {len(inconsistencies) - zero_pe_mismatches}\n")
        f.write(f"Number of missing trials: {len(missing_trials)}\n")
        
        if inconsistencies:
            f.write("\nInconsistent trial numbers:\n")
            f.write(", ".join(str(x+1) for x in inconsistencies) + "\n")
            
        if missing_trials:
            f.write("\nMissing trial numbers:\n")
            f.write(", ".join(str(x+1) for x in missing_trials) + "\n")

        print(f"Validation report saved to: {report_path}")
        return report_path, inconsistencies, missing_trials

# Example usage
if __name__ == "__main__":
    epochs_path = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.02.21\ML Classification\New Epoching\Epoch Folders\cardiac_epochs_cleaned_reref\R1_Locked-prepro_epochs-epo.fif"
    timing_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.02.07\3rd Pilot Data Analysis\Timing_analysis\trial_timing_data_cleaned.xlsx"
    output_dir = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.02.21\ML Classification\New Epoching"
    
    report_path, inconsistencies, missing_trials = validate_epochs_against_trials(epochs_path, timing_file, output_dir)
    
    # Print summary to console
    print(f"\nValidation complete!")
    print(f"Found {len(inconsistencies)} inconsistent trials")
    print(f"Found {len(missing_trials)} missing trials")
    print(f"Detailed report saved to: {report_path}")
