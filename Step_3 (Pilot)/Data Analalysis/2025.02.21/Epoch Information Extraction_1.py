import mne
import numpy as np
from datetime import datetime
import os

def inspect_epochs(epochs_path):
    """Load and inspect key information from epochs file and save to text file"""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(script_dir, f'epoch_info_detailed_{timestamp}.txt')
    
    # Load epochs
    epochs = mne.read_epochs(epochs_path)
    
    print(f"Script directory: {script_dir}")
    print(f"Report will be saved to: {output_file}")
    
    with open(output_file, 'w') as f:
        # Previous sections remain the same until Event Information
        f.write("="*50 + "\n")
        f.write("EPOCH INFORMATION REPORT (DETAILED)\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        
        # Standard information
        f.write("BASIC INFORMATION\n")
        f.write("-"*30 + "\n")
        f.write(f"Number of epochs: {len(epochs)}\n")
        f.write(f"Sampling frequency: {epochs.info['sfreq']} Hz\n")
        f.write(f"Time window: {epochs.tmin:.3f} to {epochs.tmax:.3f} seconds\n\n")
        
        # Detailed event information
        f.write("DETAILED EVENT INFORMATION\n")
        f.write("-"*30 + "\n")
        f.write("1. Event IDs:\n")
        f.write(f"{epochs.event_id}\n\n")
        
        f.write("2. Events array shape:\n")
        f.write(f"{epochs.events.shape}\n")
        f.write("Format: (n_events, [sample_nr, 0, event_id])\n\n")
        
        f.write("3. First few events:\n")
        f.write(str(epochs.events[:5]) + "\n\n")
        
        # Metadata inspection
        f.write("METADATA INSPECTION\n")
        f.write("-"*30 + "\n")
        
        if epochs.metadata is not None:
            f.write("Available metadata columns:\n")
            f.write("\n".join(epochs.metadata.columns.tolist()) + "\n\n")
            
            # Try to find PE-related columns
            pe_columns = [col for col in epochs.metadata.columns if 'pe' in col.lower()]
            if pe_columns:
                f.write("\nPE-related columns found:\n")
                f.write("\n".join(pe_columns) + "\n\n")
                
                f.write("Sample of PE values (first 5 trials):\n")
                for col in pe_columns:
                    f.write(f"\n{col}:\n")
                    f.write(str(epochs.metadata[col].head()) + "\n")
            else:
                f.write("\nNo PE-related columns found in metadata\n")
        else:
            f.write("No metadata available\n")
        
        # Check for annotation information
        f.write("\nANNOTATION INFORMATION\n")
        f.write("-"*30 + "\n")
        if hasattr(epochs, 'annotations') and epochs.annotations is not None:
            f.write("Annotations present:\n")
            f.write(str(epochs.annotations) + "\n")
        else:
            f.write("No annotations found\n")
        
        # Additional PE-specific checks
        f.write("\nPE-SPECIFIC CHECKS\n")
        f.write("-"*30 + "\n")
        description_list = epochs.event_id.keys()
        f.write("Event descriptions:\n")
        f.write("\n".join(description_list) + "\n\n")
        
        # Try to access some common PE-related attributes
        pe_attributes = ['pe_value', 'pe_sign', 'abs_pe_level', 'pe_magnitude']
        found_attributes = []
        for attr in pe_attributes:
            if hasattr(epochs, attr):
                found_attributes.append(attr)
        
        if found_attributes:
            f.write("\nFound PE-related attributes:\n")
            f.write("\n".join(found_attributes) + "\n")
        else:
            f.write("\nNo standard PE-related attributes found\n")
        
        print(f"Detailed report saved to: {output_file}")
    
    return epochs, output_file

# Example usage:
epochs_path = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.02.21\ML Classification\New Epoching\Epoch Folders\cardiac_epochs_cleaned_reref\R1_Locked-prepro_epochs-epo.fif"
epochs, report_file = inspect_epochs(epochs_path)

# Additional direct inspection
print("\nDirect inspection of epochs object:")
print("Available attributes:", dir(epochs))
if epochs.metadata is not None:
    print("\nMetadata columns:", epochs.metadata.columns.tolist())