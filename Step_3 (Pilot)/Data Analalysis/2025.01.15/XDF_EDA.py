"""
Author: Hamed Ghane
Date: January 15, 2025

Script Description:
This diagnostic tool provides a detailed exploration and visualization of XDF (Extended 
Data Format) file structures, which are commonly used in multimodal biosignal recordings. 
The script serves as a debugging and inspection utility for understanding the internal 
organization of XDF files, particularly useful when working with complex 
neurophysiological data.

Key Features and Functionality:

1. File Structure Analysis:
   - Loads XDF files and examines their internal organization
   - Identifies and counts the number of data streams
   - Provides a hierarchical view of the file's structure
   - Preserves nested relationships in the output display

2. Stream Information Display:
   - Shows detailed information for each data stream
   - Presents stream metadata in a readable, hierarchical format
   - Displays the structure of 'info' dictionaries
   - Reports time series characteristics and dimensions

3. Data Inspection:
   - Examines time series data types and shapes
   - Safely handles different data formats and structures
   - Provides preview of data content where applicable
   - Implements recursive dictionary exploration for nested structures

4. Output Format:
   - Uses indentation for clear visualization of nested structures
   - Implements safe type checking for robust operation
   - Provides clear separation between different streams
   - Includes detailed type and shape information for arrays

This tool is particularly valuable for:
- Debugging XDF file loading issues
- Understanding data organization in complex recordings
- Verifying stream configurations
- Inspecting metadata structures


Usage example:
    file_path = Path("path/to/your/xdf/file")
    explore_xdf(file_path)
"""


import pyxdf
import numpy as np
from pathlib import Path

def print_dict_structure(d, indent=0):
    """Recursively print dictionary structure"""
    for key, value in d.items():
        print(' ' * indent + f"'{key}': ", end='')
        if isinstance(value, dict):
            print()
            print_dict_structure(value, indent + 2)
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                print('[')
                for item in value:
                    print(' ' * (indent + 2) + '{')
                    print_dict_structure(item, indent + 4)
                    print(' ' * (indent + 2) + '}')
                print(' ' * indent + ']')
            else:
                print(f'list of length {len(value)}')
        else:
            print(repr(value))

def explore_xdf(file_path):
    """
    Debug exploration of XDF file structure
    """
    # Load the XDF file
    print(f"\nLoading XDF file: {file_path.name}")
    streams, fileheader = pyxdf.load_xdf(str(file_path))
    
    # Print information about each stream
    print(f"\nFound {len(streams)} streams:")
    
    for i, stream in enumerate(streams):
        print(f"\n{'='*50}")
        print(f"Stream {i + 1} Structure:")
        print(f"{'='*50}")
        
        # Print top-level keys
        print("\nTop-level keys:", list(stream.keys()))
        
        # Print info structure
        print("\nInfo structure:")
        print_dict_structure(stream['info'])
        
        # Print time series information safely
        print("\nTime series information:")
        if 'time_series' in stream:
            ts = stream['time_series']
            print(f"Type: {type(ts)}")
            if isinstance(ts, np.ndarray):
                print(f"Shape: {ts.shape}")
                print(f"Data type: {ts.dtype}")
            elif isinstance(ts, list):
                print(f"Length: {len(ts)}")
                if ts:
                    print(f"First element type: {type(ts[0])}")
                    print(f"First element preview: {str(ts[0])[:100]}...")
            else:
                print(f"Unexpected type: {type(ts)}")
        else:
            print("No time_series found")
        
        print(f"\n{'='*50}\n")

if __name__ == "__main__":
    file_path = Path(r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
    explore_xdf(file_path)
