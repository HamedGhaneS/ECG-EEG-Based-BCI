import pandas as pd
from pathlib import Path
import pyxdf
import matplotlib.pyplot as plt

def load_eeg_stream(xdf_path):
    """
    Load the EEG stream from the XDF file.

    Parameters:
    -----------
    xdf_path : str or Path
        Path to the XDF file.

    Returns:
    --------
    eeg_timestamps : list
        Timestamps of the EEG data.
    """
    streams, header = pyxdf.load_xdf(xdf_path)
    eeg_stream = streams[-1]  # Select the last stream (EEG)
    eeg_timestamps = eeg_stream['time_stamps']
    return eeg_timestamps

def load_behavioral_data(csv_path):
    """
    Load the behavioral data from the CSV file.

    Parameters:
    -----------
    csv_path : str or Path
        Path to the behavioral CSV file.

    Returns:
    --------
    start_time : float
        Start time of the behavioral data.
    end_time : float
        End time of the behavioral data.
    """
    behavioral_data = pd.read_csv(csv_path)
    start_time = behavioral_data['trial_start_time'].min()
    end_time = behavioral_data['trial_end_time'].max()
    return start_time, end_time

def check_timing_coverage(eeg_timestamps, behavioral_start, behavioral_end):
    """
    Check if the EEG recording covers the behavioral data timing and visualize durations.

    Parameters:
    ----------- 
    eeg_timestamps : list
        Timestamps of the EEG data.
    behavioral_start : float
        Start time of the behavioral data.
    behavioral_end : float
        End time of the behavioral data.

    Returns:
    --------
    None
    """
    eeg_start = eeg_timestamps[0]
    eeg_end = eeg_timestamps[-1]

    # Print timing information
    print("\nTiming Comparison:")
    print("-" * 50)
    print(f"Behavioral Data Start Time: {behavioral_start:.3f} seconds")
    print(f"Behavioral Data End Time: {behavioral_end:.3f} seconds")
    print(f"Behavioral Data Duration: {behavioral_end - behavioral_start:.3f} seconds")
    print()
    print(f"EEG Recording Start Time: {eeg_start:.3f} seconds")
    print(f"EEG Recording End Time: {eeg_end:.3f} seconds")
    print(f"EEG Recording Duration: {eeg_end - eeg_start:.3f} seconds")
    print()
    if eeg_start <= behavioral_start and eeg_end >= behavioral_end:
        print("✅ EEG recording covers the entire behavioral data collection period.")
    else:
        print("❌ EEG recording does NOT cover the entire behavioral data collection period.")
        if eeg_start > behavioral_start:
            print(f"EEG starts after the behavioral data collection by {eeg_start - behavioral_start:.3f} seconds.")
        if eeg_end < behavioral_end:
            print(f"EEG ends before the behavioral data collection by {behavioral_end - eeg_end:.3f} seconds.")
    
    # Plot durations
    plot_durations(eeg_start, eeg_end, behavioral_start, behavioral_end)

def plot_durations(eeg_start, eeg_end, behavioral_start, behavioral_end):
    """
    Plot the duration of EEG and behavioral data for comparison.

    Parameters:
    ----------- 
    eeg_start : float
        Start time of the EEG data.
    eeg_end : float
        End time of the EEG data.
    behavioral_start : float
        Start time of the behavioral data.
    behavioral_end : float
        End time of the behavioral data.

    Returns:
    --------
    None
    """
    plt.figure(figsize=(10, 5))

    # Plot EEG duration
    plt.barh(y=["EEG"], width=[eeg_end - eeg_start], left=[eeg_start], color='blue', alpha=0.7, label='EEG Duration')

    # Plot Behavioral duration
    plt.barh(y=["Behavioral"], width=[behavioral_end - behavioral_start], left=[behavioral_start], color='green', alpha=0.7, label='Behavioral Duration')

    # Add details
    plt.axvline(eeg_start, color='blue', linestyle='--', label='EEG Start')
    plt.axvline(eeg_end, color='blue', linestyle='--', label='EEG End')
    plt.axvline(behavioral_start, color='green', linestyle='--', label='Behavioral Start')
    plt.axvline(behavioral_end, color='green', linestyle='--', label='Behavioral End')

    plt.title('EEG vs Behavioral Data Duration')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Data Source')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Show plot
    plt.show()

def main():
    # Paths to the XDF file and behavioral data file
    xdf_path = Path(r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.13\Pilot\Elmira\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
    behavioral_csv_path = Path(r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.13\Pilot\Elmira\Elmira-ses001-run1-20250113-173456.csv")
    
    # Load data
    eeg_timestamps = load_eeg_stream(xdf_path)
    behavioral_start, behavioral_end = load_behavioral_data(behavioral_csv_path)
    
    # Check timing coverage
    check_timing_coverage(eeg_timestamps, behavioral_start, behavioral_end)

if __name__ == "__main__":
    main()
