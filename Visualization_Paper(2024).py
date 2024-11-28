"""
Script Title: ECG Signal Visualization and Preprocessing

Author: Hamed Ghane
Date: 2024-11-07

Description:
This script is designed to visualize and preprocess ECG signals from the study 
titled "Timing along the cardiac cycle modulates neural signals of reward-based learning,"
published in Nature Communications:
https://www.nature.com/articles/s41467-024-46921-5.

The raw data used in this script is publicly available and has been downloaded 
from the Open Science Framework (OSF):
https://osf.io/qgw7h/files/osfstorage.

Workflow:
1. Import necessary libraries including MNE for EEG/ECG data processing, 
   and SciPy and Matplotlib for filtering and plotting.
   
2. Define a bandpass filter using a low-order Butterworth design, specifically
   for real-time ECG data processing to allow for noise reduction within 0.5-40 Hz.

3. Implement a function to load subject-specific data by accessing relevant 
   event and BrainVision EEG/ECG files. It extracts sampling information and
   identifies the EEG and ECG channels available.

4. Check for the presence of the ECG channel and preprocess it with the bandpass 
   filter. Generate diagnostic plots to compare raw and filtered ECG data to 
   visualize improvements in signal clarity.

5. Provide options for the user to input the subject number and choose specific 
   analysis tasks, including plotting raw data for all channels, plotting filtered 
   ECG data, or viewing a specific ECG segment.

6. Offer an interactive menu to adjust settings, change subjects, or exit.

7. Display results and prompt for further input until the user opts to exit.

This structured workflow aims to enable efficient and interactive analysis 
of ECG data in real-time, aligning with the study's objectives of examining 
the relationship between cardiac phases and neural signal modulation.

"""

import mne
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=2):
    """Design bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def real_time_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=1000, order=2):
    """Apply a low-order bandpass filter suitable for real-time processing."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    try:
        y = lfilter(b, a, data)
    except ValueError as e:
        print(f"Filtering error: {e}")
        y = np.full_like(data, np.nan)  # Return NaNs if filtering fails
    return y

def read_subject_data(base_path, subject_number):
    """
    Read data for specified subject from the path
    """
    sub_folder = f'sub_{subject_number:02d}'
    subject_path = os.path.join(base_path, sub_folder)
    events_file = os.path.join(subject_path, 'events.mat')
    vhdr_file = os.path.join(subject_path, f'Pilot_{subject_number:02d}.vhdr')
    
    if not os.path.exists(vhdr_file):
        raise FileNotFoundError(f"No data file found for subject {subject_number}")
    
    print(f"\nReading BrainVision data for subject {subject_number}...")
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
    
    print("\nReading events file...")
    events_data = sio.loadmat(events_file)
    
    print("\nData Information:")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    duration_sec = raw.times[-1]
    print(f"Recording duration: {duration_sec:.2f} seconds")
    
    # Calculate and display the duration in hours, minutes, and seconds
    hours = int(duration_sec // 3600)
    minutes = int((duration_sec % 3600) // 60)
    seconds = int(duration_sec % 60)
    print(f"Recording duration: {hours} hour(s), {minutes} minute(s), {seconds} second(s)")
    
    eeg_channels = len(mne.pick_types(raw.info, eeg=True))
    ecg_channels = len(mne.pick_types(raw.info, ecg=True))
    print(f"\nChannel breakdown:")
    print(f"EEG channels: {eeg_channels}")
    print(f"ECG channels: {ecg_channels}")
    
    ecg_channel = 'EXT1'
    if ecg_channel in raw.ch_names:
        print(f"\nECG channel found: {ecg_channel}")
        plot_ecg_data_with_preprocessing(raw, subject_number)
    else:
        print("ECG channel (EXT1) not found!")
    
    return raw, events_data, duration_sec

def plot_ecg_data_with_preprocessing(raw, subject_number):
    """
    Plot ECG data - raw and preprocessed signals with additional diagnostics.
    """
    ecg_channel_name = 'EXT1'
    if ecg_channel_name not in raw.ch_names:
        print(f"Error: ECG channel '{ecg_channel_name}' not found in the data.")
        return

    ecg_idx = raw.ch_names.index(ecg_channel_name)
    ecg_data = raw.get_data()[ecg_idx]
    times = raw.times

    if np.isnan(ecg_data).any():
        print("Error: ECG data contains NaN values. Filtering cannot proceed.")
        return
    elif np.all(ecg_data == ecg_data[0]):
        print("Error: ECG data appears to be constant. Filtering cannot proceed.")
        return

    print(f"ECG data shape: {ecg_data.shape}")
    print(f"First 10 samples of ECG data: {ecg_data[:10]}")
    
    sampling_rate = raw.info['sfreq']
    print(f"Using sampling rate: {sampling_rate} Hz for filtering.")
    filtered_ecg = real_time_bandpass_filter(ecg_data, lowcut=0.5, highcut=40.0, fs=sampling_rate)
    
    if np.isnan(filtered_ecg).any():
        print("Error: Filtered ECG data contains NaN values. There may be a problem with the filtering process.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    ax1.plot(times, ecg_data)
    ax1.set_title(f'Raw ECG Signal - Subject {subject_number}')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    ax2.plot(times, filtered_ecg)
    ax2.set_title(f'Preprocessed ECG Signal - Subject {subject_number}')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    
    y_min = min(np.min(ecg_data), np.min(filtered_ecg))
    y_max = max(np.max(ecg_data), np.max(filtered_ecg))
    margin = (y_max - y_min) * 0.1
    ax1.set_ylim([y_min - margin, y_max + margin])
    ax2.set_ylim([y_min - margin, y_max + margin])
    
    plt.tight_layout()
    plt.show()
    
    mask_10s = times <= 10
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    ax1.plot(times[mask_10s], ecg_data[mask_10s])
    ax1.set_title(f'Raw ECG Signal (First 10 seconds) - Subject {subject_number}')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    ax2.plot(times[mask_10s], filtered_ecg[mask_10s])
    ax2.set_title(f'Preprocessed ECG Signal (First 10 seconds) - Subject {subject_number}')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    
    y_min_10s = min(np.min(ecg_data[mask_10s]), np.min(filtered_ecg[mask_10s]))
    y_max_10s = max(np.max(ecg_data[mask_10s]), np.max(filtered_ecg[mask_10s]))
    margin_10s = (y_max_10s - y_min_10s) * 0.1
    ax1.set_ylim([y_min_10s - margin_10s, y_max_10s + margin_10s])
    ax2.set_ylim([y_min_10s - margin_10s, y_max_10s + margin_10s])
    
    plt.tight_layout()
    plt.show()

def analyze_ecg_channel(raw, subject_number):
    ecg_idx = raw.ch_names.index('EXT1')
    ecg_data = raw.get_data()[ecg_idx]
    sampling_rate = raw.info['sfreq']
    
    print(f"\nECG Analysis for Subject {subject_number}:")
    print(f"ECG data shape: {ecg_data.shape}")
    print(f"Mean amplitude: {np.mean(ecg_data):.2f}")
    print(f"Max amplitude: {np.max(ecg_data):.2f}")
    print(f"Min amplitude: {np.min(ecg_data):.2f}")
    
    return ecg_data, sampling_rate

def main():
    base_path = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2024.11.05\(2024) Timing along the cardiac cycle modulates neural signals of reward-based learning"
    
    try:
        while True:
            try:
                subject_number = int(input("\nEnter subject number (1-99): "))
                if 1 <= subject_number <= 99:
                    break
                else:
                    print("Please enter a number between 1 and 99")
            except ValueError:
                print("Please enter a valid number")
        
        raw, events, duration = read_subject_data(base_path, subject_number)
        
        ecg_data, sampling_rate = analyze_ecg_channel(raw, subject_number)
        
        while True:
            print("\nPlotting options:")
            print("1: Plot all channels")
            print("2: Plot specific ECG segment")
            print("3: Change subject")
            print("4: Exit")
            
            choice = input("Enter your choice (1-4): ")
            
            if choice == '1':
                raw.plot(duration=5, n_channels=10, scalings='auto')
            elif choice == '2':
                start_time = float(input("Enter start time (seconds): "))
                duration = float(input("Enter duration (seconds): "))
                plot_ecg_segment(raw, subject_number, start_time, duration)
            elif choice == '3':
                main()
                break
            elif choice == '4':
                break
            else:
                print("Invalid choice!")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
