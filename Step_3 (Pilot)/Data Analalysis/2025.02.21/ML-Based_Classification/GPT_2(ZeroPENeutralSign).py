import mne
import numpy as np
import pandas as pd
import os
from eeg_montage import CHANNEL_NAMES

# File paths
CLEANED_DATA_PATH = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.02.10\Pilot Data Analysis\3-Eye Movement\Python\EEG_Plots\Elmira_eyeremoval_cleaned.npz"
timing_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.02.07\3rd Pilot Data Analysis\Timing_analysis\trial_timing_data_cleaned.xlsx"
rpeak_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.02.07\3rd Pilot Data Analysis\Raw Data\Behavioral-Task Data\Elmira-ses001-run1-20250131-185646_timing_pairs.CSV"
base_output_dir = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.02.21\ML Classification\New Epoching\Epoch Folders"

# Define cardiac condition
cardiac_condition = {
    "columns": ["cardiac_phase", "block_condition"],
    "events": {
        "diastole_early": 1,
        "diastole_mid": 2,
        "diastole_late": 3,
        "systole_early": 4,
        "systole_mid": 5,
        "systole_late": 6
    },
    "folder": "cardiac_epochs_cleaned_reref",
    "tmin": -0.2,
    "tmax": 0.6
}

# Create output directory
os.makedirs(os.path.join(base_output_dir, cardiac_condition["folder"]), exist_ok=True)

# Define locking event types
pe_locking_events = {
    "R1_Locked": "r1_time",
    "R2_Locked": "r2_time",
    "R3_Locked": "r3_time",
    "Outcome_Locked": "outcome_time"
}

# Load cleaned EEG data
print("Loading cleaned EEG data...")
cleaned_data = np.load(CLEANED_DATA_PATH)
eeg_data = cleaned_data['data_cleaned']
nominal_srate = cleaned_data['srate']
timestamps = cleaned_data['timestamps']

print("\nLoading timing pairs and calculating offset...")
rpeak_df = pd.read_csv(rpeak_file)
mean_offset = np.mean(rpeak_df['calculated_offset'])
print(f"Mean PC1-PC2 offset: {mean_offset:.3f}s")

# Shift timestamps for alignment with behavioral data
shifted_timestamps = timestamps - mean_offset

# Create MNE Raw object
info = mne.create_info(CHANNEL_NAMES, nominal_srate, ch_types='eeg')
raw = mne.io.RawArray(eeg_data, info)

# Preprocessing Pipeline
print("\nApplying preprocessing pipeline...")
raw.filter(0.5, 40, fir_design='firwin', verbose=False)
raw.notch_filter(50, fir_design='firwin', verbose=False)
raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)

# Load and verify timing data
print("\nLoading timing data...")
timing_data = pd.read_excel(timing_file)
print("Columns in timing file:", timing_data.columns)

# Use correct column names for PE-related variables
abs_pe_column = "abs_pe_value"
signed_pe_column = "pe_sign"

# Debugging: Check unique values in signed_pe_column before mapping
print("Unique values in signed_pe_column before mapping:", timing_data[signed_pe_column].unique())

# Correctly map SignedPE values using abs_pe_value to determine Neutral cases
timing_data["PE_sign"] = timing_data.apply(
    lambda row: "Neutral" if row[abs_pe_column] == 0 else row[signed_pe_column],
    axis=1
)

# Debugging: Check unique values after mapping
print("Unique values in signed_pe_column after mapping:", timing_data["PE_sign"].unique())

# Process cardiac condition
print("\nProcessing cardiac condition...")
for lock_name, event_column in pe_locking_events.items():
    print(f"\n{'='*50}")
    print(f"Processing {lock_name} epochs for cardiac condition...")

    # Create events and metadata
    events = []
    metadata = []
    for _, trial in timing_data.iterrows():
        cardiac = trial["cardiac_phase"].lower()
        block = trial["block_condition"].lower()
        condition = f"{cardiac}_{block}"
        event_id = cardiac_condition["events"].get(condition)

        if event_id is not None:
            sample = int((trial[event_column] - shifted_timestamps[0]) * nominal_srate)
            if 0 <= sample < len(shifted_timestamps):
                events.append([sample, 0, event_id])
                metadata.append({"AbsPE": trial[abs_pe_column], "SignedPE": trial["PE_sign"]})
    
    if len(events) > 0:
        events = np.array(events, dtype=int)
        
        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            event_id=cardiac_condition["events"],
            tmin=cardiac_condition["tmin"],
            tmax=cardiac_condition["tmax"],
            baseline=None,
            reject=None,
            preload=True,
            metadata=pd.DataFrame(metadata)  # Attach metadata
        )
        
        # Save epochs
        output_file = os.path.join(base_output_dir, cardiac_condition["folder"], f"{lock_name}-prepro_epochs-epo.fif")
        epochs.save(output_file, overwrite=True)
        print(f"Saved to: {output_file}")
    else:
        print(f"No valid events found for {lock_name}")

print("\nProcessing complete!")
