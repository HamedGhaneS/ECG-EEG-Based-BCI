import mne
import numpy as np
import pandas as pd
import pyxdf
from eeg_montage import CHANNEL_NAMES

# File paths
raw_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
timing_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.20\Pilot Data Analysis\timing_analysis\trial_timing_data_cleaned.xlsx"

# Define locking event types and corresponding timing columns
locking_events = {
    "R1_Locked": "r1_time",
    "R2_Locked": "r2_time",
    "R3_Locked": "r3_time",
    "Outcome_Locked": "outcome_time"
}

output_dir = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.21\Pilot Data Analysis"

# Load XDF data
print("Loading XDF file...")
streams, header = pyxdf.load_xdf(raw_file)

eeg_stream = streams[2]  # Assume the third stream is EEG
sfreq = float(eeg_stream['info']['nominal_srate'][0])
eeg_data = np.array(eeg_stream['time_series']).T
eeg_timestamps = np.array(eeg_stream['time_stamps'])  # Native timestamps

# Create MNE Raw object
print("Creating MNE Raw object with native timestamps...")
info = mne.create_info(CHANNEL_NAMES, sfreq, ch_types='eeg')
raw = mne.io.RawArray(eeg_data, info)

print(f"Raw data duration: {raw.times[-1]:.2f} seconds")
print(f"EEG recording start time: {eeg_timestamps[0]:.3f}s")
print(f"EEG recording end time: {eeg_timestamps[-1]:.3f}s")

# Apply preprocessing
print("Applying bandpass filter (0.5-40 Hz)...")
raw.filter(0.5, 40, fir_design='firwin', verbose=True)

print("Applying notch filter at 50 Hz...")
raw.notch_filter(50, fir_design='firwin', verbose=True)

# Load timing data from Excel
print("Loading timing data from Excel...")
timing_data = pd.read_excel(timing_file)

# Process each locking type
for lock_name, event_column in locking_events.items():
    print(f"\nProcessing {lock_name} epochs...")

    # Verify the event column exists
    if event_column not in timing_data.columns:
        print(f"Error: Column '{event_column}' not found in timing file.")
        continue

    print(f"Timing range in Excel ({lock_name}):")
    print(f"First {lock_name}: {timing_data[event_column].min():.3f}s")
    print(f"Last {lock_name}: {timing_data[event_column].max():.3f}s")

    # Create events
    events = []
    for _, trial in timing_data.iterrows():
        sample = int((trial[event_column] - eeg_timestamps[0]) * sfreq)  # Align to EEG start time
        event_id = 1 if trial['abs_pe_level'] == 'high' else 2
        events.append([sample, 0, event_id])

    events = np.array(events, dtype=int)
    events = np.sort(events, axis=0)  # Sort by time

    print(f"Created {len(events)} events")
    print(f"High PE events: {len([e for e in events if e[2] == 1])}")
    print(f"Low PE events: {len([e for e in events if e[2] == 2])}")
    print(f"Sample range: {events[0, 0]} to {events[-1, 0]}")
    print(f"Time range: {events[0, 0] / sfreq:.3f}s to {events[-1, 0] / sfreq:.3f}s")

    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id={'high_pe': 1, 'low_pe': 2},
        tmin=-0.2,    # Start 200ms before event
        tmax=0.6,     # End 600ms after event
        baseline=None, # No baseline correction
        reject=None,  # No rejection
        preload=True
    )

    print(f"\nEpochs information ({lock_name}):")
    print(f"Number of epochs: {len(epochs)}")
    print(f"Number of high PE epochs: {len(epochs['high_pe'])}")
    print(f"Number of low PE epochs: {len(epochs['low_pe'])}")

    # Save epochs
    output_file = f"{output_dir}\\{lock_name}-prepro_hep_epochs-epo.fif"
    epochs.save(output_file, overwrite=True)
    print(f"Epochs saved to {output_file}")

print("Processing complete.")
