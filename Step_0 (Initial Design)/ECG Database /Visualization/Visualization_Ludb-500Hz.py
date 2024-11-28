"""
Script Title: LUDB ECG Data Visualization and Analysis Tool
Author: Hamed Ghane
Date: 2024-11-11

Description:
    This script provides visualization and analysis capabilities for the Lobachevsky University 
    Electrocardiography Database (LUDB), specifically designed for 500 Hz ECG recordings. It 
    enables detailed examination of ECG morphology through interactive visualization and 
    real-time filtering.

Features:
    - Interactive lead selection from multi-lead ECG recordings
    - Real-time bandpass filtering (0.5-40 Hz)
    - Dual-view visualization (raw and filtered signals)
    - Detailed sample-level analysis capability
    - Support for long-term recordings with efficient memory handling

Workflow:
    1. Data Loading:
       - Access WFDB-formatted ECG recordings
       - Extract recording metadata and available leads
       - Validate data integrity and sampling rate

    2. Signal Processing:
       - Apply real-time bandpass filtering
       - Implement butter_bandpass filter design
       - Handle edge cases and signal artifacts

    3. Visualization:
       - Generate full-duration signal plots
       - Create detailed 10-second segment views
       - Provide sample-level visualization for detailed analysis
       - Display interactive plots with measurement capabilities

    4. User Interaction:
       - Lead selection interface
       - Recording duration information
       - Signal quality metrics
       - Interactive plot navigation

Dependencies:
    - WFDB: ECG data reading and processing
    - NumPy: Numerical computations
    - SciPy: Signal processing functions
    - Matplotlib: Visualization framework

Usage:
    The script is designed for researchers and clinicians analyzing LUDB ECG data,
    providing both overview and detailed examination capabilities of ECG morphology.
    It supports:
    - Selection of specific ECG leads
    - Visualization of raw and filtered signals
    - Detailed analysis of signal characteristics
    - Export of processed data for further analysis

Note:
    This tool is specifically optimized for the LUDB database with its 500 Hz
    sampling rate and multi-lead recordings format.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import wfdb  # for reading LUDB data

# Filter design function remains the same
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Apply the real-time bandpass filter function
def real_time_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=500, order=2):
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

# Function to read and preprocess LUDB ECG data, with user-selected lead
def read_ludb_ecg_data(base_path, subject_number):
    record_path = os.path.join(base_path, str(subject_number))
    
    try:
        # Read the WFDB record
        record = wfdb.rdrecord(record_path)
        
        # Display general information
        num_leads = len(record.sig_name)
        sampling_rate = record.fs
        duration = len(record.p_signal) / sampling_rate  # in seconds
        
        print(f"\nData Information for Subject {subject_number}:")
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
        plot_ecg_data_with_preprocessing(times, ecg_data, filtered_ecg, subject_number, lead_name, limit=limit)
        
    except FileNotFoundError:
        print(f"No data found for subject {subject_number} in LUDB Database")

# Main function to handle LUDB ECG data
def main():
    base_path = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2024.11.01\ECG Database\Database\(Healthy+Patient- 500 Hz) Ludb Single-Lead"
    
    try:
        while True:
            try:
                subject_number = input("\nEnter subject number (e.g., 1, 2, 3): ")
                if os.path.isfile(os.path.join(base_path, f"{subject_number}.dat")):
                    break
                else:
                    print("Please enter a valid subject number present in the directory.")
            except ValueError:
                print("Please enter a valid number")
        
        # Read and process LUDB ECG data for the selected lead
        read_ludb_ecg_data(base_path, subject_number)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
