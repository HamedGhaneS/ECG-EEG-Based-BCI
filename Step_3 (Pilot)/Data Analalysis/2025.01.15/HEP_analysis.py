import pyxdf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from pathlib import Path
from scipy import stats

def preprocess_eeg_ecg(eeg_data, fs):
    """Preprocess EEG and ECG data."""
    # Bandpass filter for EEG (0.5-40 Hz as used in the paper)
    nyq = fs / 2
    b_eeg, a_eeg = butter(1, [0.5/nyq, 40/nyq], btype='band')
    eeg_filtered = filtfilt(b_eeg, a_eeg, eeg_data, axis=0)
    
    # Bandpass filter for ECG (5-15 Hz)
    b_ecg, a_ecg = butter(1, [5/nyq, 15/nyq], btype='band')
    ecg_filtered = filtfilt(b_ecg, a_ecg, eeg_data[:, -1])
    
    return eeg_filtered, ecg_filtered

def detect_r_peaks(ecg_data, fs):
    """Detect R-peaks in ECG data."""
    peaks, _ = find_peaks(
        ecg_data,
        height=np.std(ecg_data),
        distance=int(fs * 0.5),  # Minimum 0.5s between peaks
        prominence=0.5 * np.std(ecg_data)
    )
    return peaks

def extract_hep_epochs(eeg_data, r_peaks, fs, tmin=-0.2, tmax=0.6):
    """Extract EEG epochs around R-peaks."""
    n_samples = int((tmax - tmin) * fs)
    n_channels = eeg_data.shape[1]
    n_epochs = len(r_peaks)
    epochs = np.zeros((n_epochs, n_channels, n_samples))
    
    # Time vector for plotting
    times = np.linspace(tmin, tmax, n_samples)
    
    for i, peak in enumerate(r_peaks):
        start = int(peak + tmin * fs)
        end = start + n_samples
        
        if start >= 0 and end < len(eeg_data):
            epochs[i] = eeg_data[start:end].T
            
    return epochs, times

def analyze_hep(file_path):
    """Analyze HEP from XDF file."""
    print("Loading XDF file...")
    streams, _ = pyxdf.load_xdf(str(file_path))
    eeg_stream = streams[2]  # Third stream containing EEG/ECG
    
    # Get sampling rate and data
    fs = float(eeg_stream['info']['nominal_srate'][0])
    data = eeg_stream['time_series']
    
    print("Preprocessing data...")
    # Preprocess EEG and ECG
    eeg_filtered, ecg_filtered = preprocess_eeg_ecg(data, fs)
    
    # Detect R-peaks
    print("Detecting R-peaks...")
    r_peaks = detect_r_peaks(ecg_filtered, fs)
    print(f"Found {len(r_peaks)} R-peaks")
    
    # Extract epochs
    print("Extracting epochs...")
    epochs, times = extract_hep_epochs(eeg_filtered, r_peaks, fs)
    
    # Calculate HEP (average across epochs)
    print("Calculating HEP...")
    hep = np.mean(epochs, axis=0)
    
    # Define channel groups based on the paper
    channel_groups = {
        'Frontocentral': [0, 1, 2, 3, 4, 5],  # Example indices, adjust based on your montage
        'Centroparietal': [31, 32, 33, 34, 35, 36]  # Example indices, adjust based on your montage
    }
    
    # Plot HEP for each channel group
    plt.figure(figsize=(15, 10))
    
    for i, (name, channels) in enumerate(channel_groups.items(), 1):
        plt.subplot(2, 1, i)
        
        # Calculate mean and SEM across channels
        mean_hep = np.mean(hep[channels], axis=0)
        sem_hep = stats.sem(hep[channels], axis=0)
        
        # Plot with confidence interval
        plt.plot(times * 1000, mean_hep, label=name)  # Convert to milliseconds
        plt.fill_between(times * 1000, 
                        mean_hep - sem_hep,
                        mean_hep + sem_hep,
                        alpha=0.2)
        
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.title(f'HEP - {name} channels')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (μV)')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Perform statistical analysis
    print("\nStatistical Analysis:")
    # Define time windows of interest based on the paper
    time_windows = {
        'Early': (0.1, 0.2),
        'Mid': (0.2, 0.3),
        'Late': (0.3, 0.4)
    }
    
    for window_name, (tmin, tmax) in time_windows.items():
        window_mask = (times >= tmin) & (times <= tmax)
        for group_name, channels in channel_groups.items():
            mean_amp = np.mean(hep[channels][:, window_mask])
            t_stat, p_val = stats.ttest_1samp(
                np.mean(epochs[:, channels, :][:, :, window_mask], axis=(1, 2)),
                0
            )
            print(f"\n{window_name} window ({tmin*1000:.0f}-{tmax*1000:.0f}ms) - {group_name}:")
            print(f"Mean amplitude: {mean_amp:.2f} μV")
            print(f"t-statistic: {t_stat:.2f}")
            print(f"p-value: {p_val:.4f}")
    
    return epochs, times, hep, r_peaks

if __name__ == "__main__":
    file_path = Path(r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
    epochs, times, hep, r_peaks = analyze_hep(file_path)