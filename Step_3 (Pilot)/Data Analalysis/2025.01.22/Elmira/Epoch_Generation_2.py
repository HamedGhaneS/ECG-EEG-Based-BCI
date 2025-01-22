import mne
import numpy as np
import pandas as pd
import pyxdf
from eeg_montage import CHANNEL_NAMES

# File paths
raw_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.13\Pilot\Elmira\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
timing_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.22\Pilot Data Analysis\Elmira\Trial Details\timing_analysis\trial_timing_data_cleaned.xlsx"
rpeak_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.13\Pilot\Elmira\Elmira-ses001-run1-20250113-173456_timing_pairs.CSV"
output_dir = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.22\Pilot Data Analysis\Elmira\Epoching"

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
eeg_stream = streams[3]  # Using stream 3
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
print("Unique PE levels:", timing_data['abs_pe_level'].unique())

# Process each locking type
for lock_name, event_column in locking_events.items():
    print(f"\n{'='*50}")
    print(f"Processing {lock_name} epochs...")
    
    # Create events
    events = []
    for _, trial in timing_data.iterrows():
        sample = int((trial[event_column] - shifted_timestamps[0]) * nominal_srate)
        event_id = 1 if trial['abs_pe_level'].lower() == 'high' else 2
        events.append([sample, 0, event_id])
    
    events = np.array(events, dtype=int)
    events = np.sort(events, axis=0)
    
    print(f"\nEvent information for {lock_name}:")
    print(f"Total events: {len(events)}")
    print(f"High PE events: {len([e for e in events if e[2] == 1])}")
    print(f"Low PE events: {len([e for e in events if e[2] == 2])}")
    print(f"Time range: {events[0, 0]/nominal_srate:.3f}s to {events[-1, 0]/nominal_srate:.3f}s")
    
    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id={'high_pe': 1, 'low_pe': 2},
        tmin=-0.2,    
        tmax=0.6,     
        baseline=None,
        reject=None,  
        preload=True
    )
    
    print(f"\nEpochs created for {lock_name}:")
    print(f"Total epochs: {len(epochs)}")
    print(f"High PE epochs: {len(epochs['high_pe'])}")
    print(f"Low PE epochs: {len(epochs['low_pe'])}")
    
    # Save epochs
    output_file = f"{output_dir}\\{lock_name}-prepro_hep_epochs-epo.fif"
    epochs.save(output_file, overwrite=True)
    print(f"Saved to: {output_file}")

print("\nProcessing complete!")
print("="*50)
