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