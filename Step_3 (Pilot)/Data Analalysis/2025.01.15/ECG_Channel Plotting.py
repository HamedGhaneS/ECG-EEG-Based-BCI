import pyxdf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_last_channel(file_path, seconds=10):
    """Plot the first few seconds of the last channel from the EEG data."""
    
    # Load the XDF file
    print("Loading XDF file...")
    streams, fileheader = pyxdf.load_xdf(str(file_path))
    
    # Find the EEG stream (stream 3 based on the previous output)
    eeg_stream = streams[2]  # Index 2 for the third stream
    
    # Get sampling rate
    fs = float(eeg_stream['info']['nominal_srate'][0])
    print(f"Sampling rate: {fs} Hz")
    
    # Get the data
    data = eeg_stream['time_series']
    
    # Extract last channel
    last_channel = data[:, -1]
    
    # Calculate number of samples for the requested duration
    n_samples = int(seconds * fs)
    
    # Create time axis
    time = np.arange(n_samples) / fs
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(time, last_channel[:n_samples], 'b-', linewidth=1)
    plt.title('Last Channel (Potential ECG) - First 10 Seconds')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (μV)')
    plt.grid(True)
    
    # Add some basic statistics
    print(f"\nChannel Statistics:")
    print(f"Mean: {np.mean(last_channel):.2f}")
    print(f"Std: {np.std(last_channel):.2f}")
    print(f"Min: {np.min(last_channel):.2f}")
    print(f"Max: {np.max(last_channel):.2f}")
    
    # Try to estimate heart rate from peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(last_channel[:n_samples], distance=int(fs*0.5))  # Minimum 0.5s between peaks
    if len(peaks) > 1:
        mean_hr = 60 / np.mean(np.diff(peaks) / fs)
        print(f"\nEstimated heart rate: {mean_hr:.1f} BPM")
    
    plt.show()

if __name__ == "__main__":
    file_path = Path(r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
    plot_last_channel(file_path)