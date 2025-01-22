import pyxdf
from pathlib import Path
import numpy as np

def extract_xdf_info(xdf_path):
    """
    Extract and display detailed information from an XDF file.

    Parameters:
    -----------
    xdf_path : str or Path
        Path to the XDF file.
    """
    # Load the XDF file
    print(f"Loading XDF file: {xdf_path}")
    try:
        streams, header = pyxdf.load_xdf(xdf_path)
    except Exception as e:
        print(f"Error loading XDF file: {e}")
        return

    print("\nXDF File Information:")
    print("-" * 50)
    
    # Iterate through all streams
    for i, stream in enumerate(streams):
        print(f"\nStream {i + 1}:")
        print("-" * 20)
        print(f"Name: {stream['info']['name'][0]}")
        print(f"Type: {stream['info']['type'][0]}")
        print(f"Nominal Sampling Rate: {stream['info']['nominal_srate'][0]} Hz")
        print(f"Number of Channels: {stream['info']['channel_count'][0]}")
        
        # Handle stream ID correctly
        stream_id = stream['info'].get('stream_id', None)
        if isinstance(stream_id, int):
            print(f"Stream ID: {stream_id}")
        elif isinstance(stream_id, list) and len(stream_id) > 0:
            print(f"Stream ID: {stream_id[0]}")
        else:
            print("Stream ID: Not Available")

        print(f"Start Time: {stream['info'].get('created_at', ['N/A'])[0]}")

        # Check the data type and display size
        time_series = stream['time_series']
        time_stamps = stream['time_stamps']
        
        print(f"Time Series Data Type: {type(time_series)}")
        print(f"Number of Samples: {len(time_series)}")
        if isinstance(time_series, list):
            print(f"Time Series Length (first sample): {len(time_series[0]) if time_series else 'N/A'}")
        elif isinstance(time_series, np.ndarray):
            print(f"Data Shape: {time_series.shape}")
        
        print(f"Time Stamps Data Type: {type(time_stamps)}")
        print(f"Number of Time Stamps: {len(time_stamps)}")

        # Print a small sample of the data
        print("\nExample Data (First 5 Samples):")
        print(time_series[:5])
        print("\nExample Timestamps (First 5):")
        print(time_stamps[:5])

    print("\nXDF File Inspection Completed.")

# Path to the malfunctioned XDF file
xdf_path = Path(r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf")

# Extract and display information
extract_xdf_info(xdf_path)
