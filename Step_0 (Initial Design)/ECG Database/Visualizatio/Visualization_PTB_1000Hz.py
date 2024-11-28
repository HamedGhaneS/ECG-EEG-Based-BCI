import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import wfdb  # for reading ECG data in the PTB format

# Filter design function remains the same
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Apply the real-time bandpass filter function
def real_time_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=1000, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    try:
        y = lfilter(b, a, data)
    except ValueError as e:
        print(f"Filtering error: {e}")
        y = np.full_like(data, np.nan)
    return y

# Function to plot both raw and filtered ECG signals
def plot_ecg_data_with_preprocessing(times, ecg_data, filtered_ecg, subject_number, lead_name, limit=600):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot full duration (limited to 60 minutes or as specified by `limit`)
    ax1.plot(times[:limit], ecg_data[:limit])
    ax1.set_title(f'Raw ECG Signal (Limited to {limit / 60:.2f} minutes) - Subject {subject_number}, Lead {lead_name}')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    ax2.plot(times[:limit], filtered_ecg[:limit])
    ax2.set_title(f'Preprocessed ECG Signal (Limited to {limit / 60:.2f} minutes) - Subject {subject_number}, Lead {lead_name}')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot initial 10 seconds with markers for each sample
    mask_10s = times <= 10
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Raw ECG with markers
    ax1.plot(times[mask_10s], ecg_data[mask_10s], label='Raw ECG')
    ax1.scatter(times[mask_10s], ecg_data[mask_10s], color='blue', s=10, label='Samples')
    ax1.set_title(f'Raw ECG Signal (First 10 seconds) - Subject {subject_number}, Lead {lead_name}')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax1.legend()

    # Preprocessed ECG with markers
    ax2.plot(times[mask_10s], filtered_ecg[mask_10s], label='Filtered ECG')
    ax2.scatter(times[mask_10s], filtered_ecg[mask_10s], color='orange', s=10, label='Samples')
    ax2.set_title(f'Preprocessed ECG Signal (First 10 seconds) - Subject {subject_number}, Lead {lead_name}')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Function to read and preprocess PTB ECG data, with user-selected lead
def read_ptb_ecg_data(base_path, subject_folder):
    folder_path = os.path.join(base_path, subject_folder)
    
    # Look for .dat and .hea files in the folder
    dat_files = [f for f in os.listdir(folder_path) if f.endswith('.dat')]
    if not dat_files:
        print(f"No .dat files found in {folder_path}")
        return
    
    # Assume the first .dat file for reading (could prompt user if multiple .dat files are available)
    record_name = dat_files[0].replace('.dat', '')
    record_path = os.path.join(folder_path, record_name)
    
    try:
        # Read the WFDB record
        record = wfdb.rdrecord(record_path)
        
        # Display general information
        num_leads = len(record.sig_name)
        sampling_rate = record.fs
        duration = len(record.p_signal) / sampling_rate  # in seconds
        
        print(f"\nData Information for Subject {subject_folder}:")
        print(f"Number of leads: {num_leads}")
        print(f"Available leads: {', '.join(record.sig_name)}")
        print(f"Sampling rate: {sampling_rate} Hz")
        print(f"Recording duration: {duration / 60:.2f} minutes ({duration} seconds)")
        
        # Ask the user to select a lead to plot
        while True:
            lead_name = input(f"Enter the lead name to plot from the available options ({', '.join(record.sig_name)}): ")
            if lead_name in record.sig_name:
                break
            else:
                print("Invalid lead name. Please select a lead from the available options.")
        
        # Get the index of the selected lead
        lead_idx = record.sig_name.index(lead_name)
        ecg_data = record.p_signal[:, lead_idx]
        times = np.arange(len(ecg_data)) / sampling_rate
        
        # Filter ECG data
        filtered_ecg = real_time_bandpass_filter(ecg_data, lowcut=0.5, highcut=40.0, fs=sampling_rate)
        
        # Limit duration to 60 minutes for full plot (convert to samples)
        limit = int(min(60 * 60 * sampling_rate, len(ecg_data)))
        
        # Plot the ECG data for the selected lead
        plot_ecg_data_with_preprocessing(times, ecg_data, filtered_ecg, subject_folder, lead_name, limit=limit)
        
    except FileNotFoundError:
        print(f"No data found for subject {subject_folder} in PTB Database")

# Main function to handle PTB ECG data
def main():
    base_path = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2024.11.01\ECG Database\Database\(Patient+Healthy - 1000 Hz) PTB Diagnostic Database"
    
    try:
        # List all subdirectories (each representing a patient) in the base path
        patient_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        if not patient_folders:
            print("No patient folders found in the directory.")
            return
        
        print("\nAvailable Patient Folders for Visualization:")
        for i, folder_name in enumerate(patient_folders):
            print(f"{i + 1}: {folder_name}")
        
        # Ask the user to select a patient folder
        while True:
            try:
                choice = int(input("\nSelect the patient folder number to visualize: "))
                if 1 <= choice <= len(patient_folders):
                    subject_folder = patient_folders[choice - 1]
                    break
                else:
                    print("Please enter a valid folder number.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # Read and process PTB ECG data for the selected lead
        read_ptb_ecg_data(base_path, subject_folder)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
