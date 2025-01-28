# =============================================================================
# Prepared by: Hamed Ghane
# Date: 28th January 2025
# =============================================================================
# Script Overview:
# This script performs preprocessing of EEG data and generates epochs based on 
# defined conditions and locking events. Below is a detailed breakdown of the 
# workflow and its components.

# =============================================================================
# 1. Preprocessing Steps
# -----------------------------------------------------------------------------
# - Data Loading:
#   - EEG data is loaded from an XDF file located at:
#     -> raw_file: H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.13\Pilot\Elmira\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf
#   - Timing data is loaded from:
#     -> timing_file: H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.22\Pilot Data Analysis\Elmira\Trial Details\timing_analysis\trial_timing_data_cleaned.xlsx
#   - R-peak timing data is loaded from:
#     -> rpeak_file: H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.13\Pilot\Elmira\Elmira-ses001-run1-20250113-173456_timing_pairs.CSV
#   - The calculated mean offset from the R-peak data is applied for timestamp adjustments.

# - Referencing:
#   - An average reference is applied to all EEG channels:
#     -> raw.set_eeg_reference('average', projection=True)

# - Filtering:
#   - Band-pass filter: 0.5–40 Hz
#   - Notch filter: 50 Hz (to remove powerline noise)

# - Data Verification:
#   - Dimensions of EEG data, sampling rate, and recording duration are verified.

# =============================================================================
# 2. Event Handling
# -----------------------------------------------------------------------------
# - Defined Conditions:
#   - **abs_pe**:
#     -> Column: "abs_pe_level"
#     -> Events: {"high": 1, "low": 2}
#     -> Output folder: "abs_pe_epochs"
#   -**pe_sign**:
#     -> Column: "pe_value"
#     -> Events: {"positive": 1, "negative": 2}
#     -> Output folder: "pe_sign_epochs"

# - Locking Events:
#   - Events are locked to the following time points:
#     -> "R1_Locked": "r1_time"
#     -> "R2_Locked": "r2_time"
#     -> "R3_Locked": "r3_time"
#     -> "Outcome_Locked": "outcome_time"

# - Event Categorization:
#   - Events are created for each trial based on the corresponding condition 
#     and locking time.
#   - Events with undefined or zero prediction errors are excluded for the 
#     "pe_sign" condition.

# =============================================================================
# 3. Epoch Generation
# -----------------------------------------------------------------------------
# - Event Mapping:
#   - Events are mapped to sample indices using adjusted timestamps and the 
#     nominal sampling rate.

# - Epoch Creation:
#   - Epoch time window: -0.2s to 0.6s
#   - Baseline correction: Deferred for later analysis
#   - Epochs are created for each condition and locking type.

# - Output Storage:
#   - Epochs are saved in the following structure:
#     -> Base directory: H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.28\Pilot Data Analysis\Enhanced Preprocessing\Epoch Generation
#     -> Subfolders: "abs_pe_epochs", "feedback_epochs", "pe_sign_epochs"
#     -> File format: "{lock_name}-prepro_epochs-epo.fif"

# =============================================================================
# 4. Output and Reporting
# -----------------------------------------------------------------------------
# - Event Statistics:
#   - Total number of events and their distribution by type are reported for 
#     each locking type and condition.

# - Epoch Details:
#   - Total number of epochs and their distribution across event categories 
#     are printed.

# =============================================================================
# Summary:
# This workflow ensures precise preprocessing and epoch generation with robust 
# data organization for further analysis.
# =============================================================================


import mne
import numpy as np
import pandas as pd
import pyxdf
import os
from eeg_montage import CHANNEL_NAMES

while True:    
    # Ask user for participant name
    participant = input("Enter participant name (e.g., Elmira, Harry) or press 'q' to exit: ").strip()

    if participant == "q":
        print("Exiting the program.")
        break

    # Define file paths based on participant name
    if participant == "Elmira":
        raw_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.13\Pilot\Elmira\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
        timing_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.22\Pilot Data Analysis\Elmira\Trial Details\timing_analysis\trial_timing_data_cleaned.xlsx"
        rpeak_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.13\Pilot\Elmira\Elmira-ses001-run1-20250113-173456_timing_pairs.CSV"
        base_output_dir = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.28\Pilot Data Analysis\Enhanced Epoching\Elmira"
        eeg_stream_index = 3
    elif participant == "Harry":
        raw_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
        timing_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.20\Pilot Data Analysis\timing_analysis\trial_timing_data_cleaned.xlsx"
        rpeak_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry\Harry-ses001-run1-20250114-130445_timing_pairs.CSV"
        base_output_dir = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.28\Pilot Data Analysis\Enhanced Epoching\Harry"
        eeg_stream_index = 2
    else:
        print(f"Unknown participant: {participant}. Please try again.")
        continue

    # Define conditions and their corresponding folders
    conditions = {
        "abs_pe": {
            "column": "abs_pe_level",
            "events": {"high": 1, "low": 2},
            "folder": "abs_pe_epochs"
        },
        "pe_sign": {
            "column": "pe_value",
            "events": {"positive": 1, "negative": 2},
            "folder": "pe_sign_epochs"
        }
    }

    # Create output directories for each condition to store generated epochs
    for condition in conditions.values():
        os.makedirs(os.path.join(base_output_dir, condition["folder"]), exist_ok=True)

    # Define locking event types for epoch alignment
    locking_events = {
        "R1_Locked": "r1_time",       # Locking event at R1
        "R2_Locked": "r2_time",       # Locking event at R2
        "R3_Locked": "r3_time",       # Locking event at R3
        "Outcome_Locked": "outcome_time"  # Locking event at the outcome
    }

    # Load and verify data from XDF file
    print("Loading XDF file...")
    streams, header = pyxdf.load_xdf(raw_file)

    # Log number of streams loaded from the XDF file
    print("\nStreams loaded. Verifying structure...")
    print(f"Number of streams: {len(streams)}")

    # Load timing pairs and calculate offset for timestamp adjustment
    print("\nLoading timing pairs and calculating offset...")
    rpeak_df = pd.read_csv(rpeak_file)
    mean_offset = np.mean(rpeak_df['calculated_offset'])
    print(f"Mean PC1-PC2 offset: {mean_offset:.3f}s")

    # Extract EEG data and relevant information
    print("\nExtracting EEG data...")
    if participant == "Elmira":
        eeg_stream = streams[3]  # Select the appropriate EEG stream (index 3 for Elmira)
    elif participant == "Harry":
        eeg_stream = streams[2]  # Select the appropriate EEG stream (index 2 for Harry) 
    else:
        raise ValueError(f"Unknown participant: {participant}") 

    nominal_srate = float(eeg_stream['info']['nominal_srate'][0])  # Sampling rate
    eeg_data = np.array(eeg_stream['time_series']).T  # EEG time-series data
    original_timestamps = np.array(eeg_stream['time_stamps'])  # Original timestamps

    # Log basic information about the EEG data
    print("\nVerifying data dimensions:")
    print(f"EEG data shape: {eeg_data.shape}")
    print(f"Sampling rate: {nominal_srate} Hz")
    print(f"Recording duration: {original_timestamps[-1] - original_timestamps[0]:.2f}s")

    # Adjust timestamps using the calculated offset to align with timing data
    print("\nShifting timestamps by calculated offset...")
    shifted_timestamps = original_timestamps - mean_offset
    print(f"Shifted timestamp range: {shifted_timestamps[0]:.3f}s to {shifted_timestamps[-1]:.3f}s")

    # Create MNE Raw object for EEG data representation
    print("\nCreating MNE Raw object...")
    info = mne.create_info(CHANNEL_NAMES, nominal_srate, ch_types='eeg')  # Create info structure
    raw = mne.io.RawArray(eeg_data, info)  # Create Raw object with EEG data

    # Preprocessing steps
    print("\nApplying preprocessing...")

    # Apply average reference to the EEG data
    print("Applying average reference...")
    raw.set_eeg_reference('average', projection=True)  # Set average reference
    raw.apply_proj()  # Apply reference projection
    print("Average reference applied.")

    # Apply band-pass filter to retain relevant frequency range
    print("Applying band-pass filter (0.5-40 Hz)...")
    raw.filter(0.5, 40, fir_design='firwin', verbose=False)  # Band-pass filter
    print("Band-pass filter applied.")

    # Apply notch filter to remove powerline noise at 50 Hz
    print("Applying notch filter (50 Hz)...")
    raw.notch_filter(50, fir_design='firwin', verbose=False)  # Notch filter
    print("Notch filter applied.")

    # Load timing data for trial-specific information
    print("\nLoading timing data...")
    timing_data = pd.read_excel(timing_file)  # Load timing data from Excel file
    print(f"Number of trials: {len(timing_data)}")

    # Function to determine event ID based on trial and condition
    # This function maps trial data to specific event IDs for epoching
    def get_event_id(trial, condition_info):
        if condition_info["column"] == "pe_value":  # For signed prediction error (PE)
            if trial[condition_info["column"]] > 0:
                return condition_info["events"]["positive"]
            elif trial[condition_info["column"]] < 0:
                return condition_info["events"]["negative"]
            else:
                return None  # Skip trials with zero PE
        else:  # For absolute PE level
            return condition_info["events"]["high"] if trial[condition_info["column"]].lower() == "high" else condition_info["events"]["low"]

    # Process each condition and locking type to create epochs
    for condition_name, condition_info in conditions.items():
        print(f"\nProcessing {condition_name} condition...")
        
        for lock_name, event_column in locking_events.items():
            print(f"\n{'='*50}")
            print(f"Processing {lock_name} epochs for {condition_name}...")
            
            # Create event list for the current condition and locking event
            events = []
            print("Creating events...")
            for _, trial in timing_data.iterrows():
                event_id = get_event_id(trial, condition_info)  # Get event ID for the trial
                if event_id is not None:  # Skip invalid trials
                    # Calculate sample index based on the locking event timestamp
                    sample = int((trial[event_column] - shifted_timestamps[0]) * nominal_srate)
                    events.append([sample, 0, event_id])  # Append event to the list
            
            # Convert events to numpy array and sort by time
            events = np.array(events, dtype=int)
            events = np.sort(events, axis=0)
            
            # Create event_id dictionary based on the condition
            if condition_name == "pe_sign":
                event_id = {'positive_pe': 1, 'negative_pe': 2}
            else:
                event_id = {'high_pe': 1, 'low_pe': 2}
            
            # Log event statistics for debugging
            print(f"\nEvent information for {lock_name}:")
            print(f"Total events: {len(events)}")
            print(f"Events of type 1: {len([e for e in events if e[2] == 1])}")
            print(f"Events of type 2: {len([e for e in events if e[2] == 2])}")
            print(f"Time range: {events[0, 0]/nominal_srate:.3f}s to {events[-1, 0]/nominal_srate:.3f}s")
            
            # Create epochs using MNE's Epochs object
            print("Creating epochs...")
            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_id,
                tmin=-0.2,    # Start time of epochs (200 ms before event)
                tmax=0.6,     # End time of epochs (600 ms after event)
                baseline=None,  # No baseline correction applied
                reject=None,  # No rejection criteria applied
                preload=True  # Preload data into memory
            )
            print(f"Epochs created for {lock_name} ({condition_name}).")
            
            # Log detailed epoch statistics
            print(f"\nEpochs details:")
            print(f"Total epochs: {len(epochs)}")
            for event_name, event_num in event_id.items():
                print(f"{event_name} epochs: {len(epochs[event_name])}")
            
            # Save generated epochs to the specified folder
            output_file = os.path.join(base_output_dir, condition_info["folder"], 
                                    f"{lock_name}-prepro_epochs-epo.fif")
            epochs.save(output_file, overwrite=True)  # Save epochs
            print(f"Epochs saved to: {output_file}")

    print("\nProcessing complete!")
    print("="*25)
    print("Epoch generation and preprocessing finished.")
    print("="*50)
print ("Exiting the program")
