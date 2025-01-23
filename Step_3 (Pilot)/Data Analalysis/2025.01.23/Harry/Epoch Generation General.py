import mne
import numpy as np
import pandas as pd
import pyxdf
import os
from eeg_montage import CHANNEL_NAMES

# File paths
raw_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
timing_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.20\Pilot Data Analysis\timing_analysis\trial_timing_data_cleaned.xlsx"
rpeak_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry\Harry-ses001-run1-20250114-130445_timing_pairs.CSV"
base_output_dir = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.23\Pilot Data Analysis\Harry\Epoching"

# Define conditions and their corresponding folders
conditions = {
    "abs_pe": {
        "column": "abs_pe_level",
        "events": {"high": 1, "low": 2},
        "folder": "abs_pe_epochs"
    },
    "feedback": {
        "column": "feedback",
        "events": {"win": 1, "loss": 2},
        "folder": "feedback_epochs"
    },
    "pe_sign": {
        "column": "pe_value",
        "events": {"positive": 1, "negative": 2},
        "folder": "pe_sign_epochs"
    }
}

# Create output directories
for condition in conditions.values():
    os.makedirs(os.path.join(base_output_dir, condition["folder"]), exist_ok=True)

# Define locking event types
locking_events = {
    "R1_Locked": "r1_time",
    "R2_Locked": "r2_time",
    "R3_Locked": "r3_time",
    "Outcome_Locked": "outcome_time"
}

# Load and verify data
print("Loading XDF file...")
streams, header = pyxdf.load_xdf(raw_file)

print("\nLoading timing pairs and calculating offset...")
rpeak_df = pd.read_csv(rpeak_file)
mean_offset = np.mean(rpeak_df['calculated_offset'])
print(f"Mean PC1-PC2 offset: {mean_offset:.3f}s")

# Get EEG data
eeg_stream = streams[2]  # Using stream 2
nominal_srate = float(eeg_stream['info']['nominal_srate'][0])
eeg_data = np.array(eeg_stream['time_series']).T
original_timestamps = np.array(eeg_stream['time_stamps'])

# Print data verification
print("\nVerifying data dimensions:")
print(f"EEG data shape: {eeg_data.shape}")
print(f"Sampling rate: {nominal_srate} Hz")
print(f"Recording duration: {original_timestamps[-1] - original_timestamps[0]:.2f}s")

# Shift timestamps and create MNE Raw object
shifted_timestamps = original_timestamps - mean_offset
print(f"\nShifted timestamp range: {shifted_timestamps[0]:.3f}s to {shifted_timestamps[-1]:.3f}s")

info = mne.create_info(CHANNEL_NAMES, nominal_srate, ch_types='eeg')
raw = mne.io.RawArray(eeg_data, info)

# Preprocessing
print("\nApplying preprocessing...")
raw.filter(0.5, 40, fir_design='firwin', verbose=False)
raw.notch_filter(50, fir_design='firwin', verbose=False)

# Load and verify timing data
print("\nLoading timing data...")
timing_data = pd.read_excel(timing_file)
print(f"Number of trials: {len(timing_data)}")

# Function to determine event ID based on condition
def get_event_id(trial, condition_info):
    if condition_info["column"] == "pe_value":
        if trial[condition_info["column"]] > 0:
            return condition_info["events"]["positive"]
        elif trial[condition_info["column"]] < 0:
            return condition_info["events"]["negative"]
        else:
            return None
    elif condition_info["column"] == "feedback":
        return condition_info["events"]["win"] if trial[condition_info["column"]].lower() == "win" else condition_info["events"]["loss"]
    else:  # abs_pe_level
        return condition_info["events"]["high"] if trial[condition_info["column"]].lower() == "high" else condition_info["events"]["low"]

# Process each condition and locking type
for condition_name, condition_info in conditions.items():
    print(f"\nProcessing {condition_name} condition...")
    
    for lock_name, event_column in locking_events.items():
        print(f"\n{'='*50}")
        print(f"Processing {lock_name} epochs for {condition_name}...")
        
        # Create events
        events = []
        for _, trial in timing_data.iterrows():
            event_id = get_event_id(trial, condition_info)
            if event_id is not None:  # Skip trials with zero PE for pe_sign condition
                sample = int((trial[event_column] - shifted_timestamps[0]) * nominal_srate)
                events.append([sample, 0, event_id])
        
        events = np.array(events, dtype=int)
        events = np.sort(events, axis=0)
        
        # Create event_id dictionary based on condition
        if condition_name == "pe_sign":
            event_id = {'positive_pe': 1, 'negative_pe': 2}
        elif condition_name == "feedback":
            event_id = {'win': 1, 'loss': 2}
        else:
            event_id = {'high_pe': 1, 'low_pe': 2}
        
        print(f"\nEvent information for {lock_name}:")
        print(f"Total events: {len(events)}")
        print(f"Events of type 1: {len([e for e in events if e[2] == 1])}")
        print(f"Events of type 2: {len([e for e in events if e[2] == 2])}")
        print(f"Time range: {events[0, 0]/nominal_srate:.3f}s to {events[-1, 0]/nominal_srate:.3f}s")
        
        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=-0.2,    
            tmax=0.6,     
            baseline=None,
            reject=None,  
            preload=True
        )
        
        # Print epoch information
        print(f"\nEpochs created for {lock_name} ({condition_name}):")
        print(f"Total epochs: {len(epochs)}")
        for event_name, event_num in event_id.items():
            print(f"{event_name} epochs: {len(epochs[event_name])}")
        
        # Save epochs in condition-specific folder
        output_file = os.path.join(base_output_dir, condition_info["folder"], 
                                 f"{lock_name}-prepro_epochs-epo.fif")
        epochs.save(output_file, overwrite=True)
        print(f"Saved to: {output_file}")

print("\nProcessing complete!")
print("="*50)