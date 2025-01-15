"""
XDF Channel Structure Explorer
Author: Hamed Ghane
Date: January 15, 2025

This script explores and displays channel information from XDF files,
specifically designed for BrainProducts EEG recordings.
"""

import pyxdf
from pathlib import Path

def print_channel_info(stream):
    """Print detailed channel information from an XDF stream.
    
    Args:
        stream (dict): XDF stream dictionary containing recording information
    """
    try:
        if 'desc' in stream['info']:
            desc = stream['info']['desc'][0]
            if isinstance(desc, dict) and 'channels' in desc:
                channels = desc['channels'][0]['channel']
                print("\nFound channel labels:")
                for i, channel in enumerate(channels):
                    if 'label' in channel:
                        print(f"Channel {i+1}: {channel['label'][0]}")
                    else:
                        print(f"Channel {i+1}: No label found")
            else:
                print("No detailed channel information available in desc")
        else:
            print("No desc field found in stream info")
            
    except Exception as e:
        print(f"Note: Could not process channel information: {str(e)}")
        print("Available info keys:", list(stream['info'].keys()))

def print_stream_info(stream):
    """Print general information about a stream.
    
    Args:
        stream (dict): XDF stream dictionary
    """
    try:
        print(f"Name: {stream['info'].get('name', ['Unknown'])[0]}")
        print(f"Type: {stream['info'].get('type', ['Unknown'])[0]}")
        print(f"Channel count: {stream['info'].get('channel_count', ['Unknown'])[0]}")
        
        if 'nominal_srate' in stream['info']:
            print(f"Sampling rate: {stream['info']['nominal_srate'][0]} Hz")
        
        # Print additional metadata if available
        if 'desc' in stream['info']:
            desc = stream['info']['desc'][0]
            if isinstance(desc, dict):
                print("\nAdditional metadata:")
                for key in desc.keys():
                    if key != 'channels':  # We handle channels separately
                        print(f"  {key}")
                        
    except Exception as e:
        print(f"Error printing stream info: {str(e)}")

def explore_xdf_structure(file_path):
    """Explore and print the structure of an XDF file.
    
    Args:
        file_path (Path): Path to the XDF file
    """
    try:
        print(f"Loading XDF file: {file_path}")
        streams, header = pyxdf.load_xdf(str(file_path))
        
        print(f"\nFound {len(streams)} streams in the recording")
        
        for i, stream in enumerate(streams):
            print(f"\nExamining Stream {i}")
            print("-" * 50)
            
            # Print basic stream information
            print_stream_info(stream)
            
            # Print time series information if available
            if 'time_series' in stream:
                ts = stream['time_series']
                print("\nTime series information:")
                print(f"Type: {type(ts)}")
                if hasattr(ts, 'shape'):
                    print(f"Shape: {ts.shape}")
                    print(f"Data type: {ts.dtype}")
                elif isinstance(ts, list):
                    print(f"Length: {len(ts)}")
                    if ts:
                        print(f"First element type: {type(ts[0])}")
            
            # Print channel information
            print_channel_info(stream)
            
            print("-" * 50)
            
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except Exception as e:
        print(f"Error processing XDF file: {str(e)}")

def main():
    # Define the path to your XDF file
    file_path = Path(r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
    
    # Run the exploration
    explore_xdf_structure(file_path)

if __name__ == "__main__":
    main()
