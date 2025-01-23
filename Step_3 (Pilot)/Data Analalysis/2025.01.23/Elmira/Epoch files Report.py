import mne
import os

def extract_fif_info(file_path):
    try:
        # Read the epochs file
        epochs = mne.read_epochs(file_path, preload=False)

        # Extract information
        info = {
            "File name": os.path.basename(file_path),
            "Number of epochs": len(epochs),
            "Number of channels": len(epochs.info['ch_names']),
            "Channel names": epochs.info['ch_names'],
            "Sampling frequency (Hz)": epochs.info['sfreq'],
            "Epoch length (s)": epochs.times[-1] - epochs.times[0],
            "Events": epochs.event_id,
            # Add counts for each event type
            "Epochs per event": {event_name: len(epochs[event_name]) for event_name in epochs.event_id.keys()}
        }

        return info

    except Exception as e:
        return {"File name": os.path.basename(file_path), "Error": str(e)}

def generate_structured_report(base_directory):
    # Define condition folders
    conditions = ["abs_pe_epochs", "feedback_epochs", "pe_sign_epochs"]
    
    # Create a report file in the base directory
    report_file = os.path.join(base_directory, "complete_fif_report.txt")
    
    with open(report_file, "w") as f:
        # Write report header
        f.write("="*80 + "\n")
        f.write("COMPLETE EPOCHS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Process each condition folder
        for condition in conditions:
            condition_dir = os.path.join(base_directory, condition)
            
            if not os.path.exists(condition_dir):
                f.write(f"\nWarning: Directory not found - {condition}\n")
                continue
                
            # Write condition header
            f.write("\n" + "="*80 + "\n")
            f.write(f"CONDITION: {condition}\n")
            f.write("="*80 + "\n\n")
            
            # Process each fif file in the condition folder
            fif_files = [f for f in os.listdir(condition_dir) if f.endswith('.fif')]
            fif_files.sort()  # Sort files for consistent ordering
            
            for fif_file in fif_files:
                file_path = os.path.join(condition_dir, fif_file)
                file_info = extract_fif_info(file_path)
                
                # Write file information
                f.write("-"*40 + "\n")
                f.write(f"Lock type: {fif_file.split('-')[0]}\n")
                f.write("-"*40 + "\n")
                
                for key, value in file_info.items():
                    if key == "Channel names":
                        f.write(f"{key}: [total: {len(value)}]\n")
                    elif key == "Epochs per event":
                        f.write(f"{key}:\n")
                        for event_name, count in value.items():
                            f.write(f"  - {event_name}: {count}\n")
                    elif key != "File name":  # Skip file name as we're using lock type instead
                        f.write(f"{key}: {value}\n")
                f.write("\n")

        # Write report footer
        f.write("\n" + "="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")

    print(f"Complete report generated: {report_file}")

# Base directory containing the condition folders
base_output_dir = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.23\Pilot Data Analysis\Elmira\Epoching"
generate_structured_report(base_output_dir)