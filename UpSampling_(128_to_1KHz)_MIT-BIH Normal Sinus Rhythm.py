import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
import wfdb  # for reading MIT-BIH data
import os

# Function to resample ECG data from 128 Hz to 1 kHz
def resample_ecg_signal(ecg_data, original_rate=128, target_rate=1000):
    num_samples = int(len(ecg_data) * (target_rate / original_rate))
    resampled_ecg = resample(ecg_data, num_samples)
    duration = len(ecg_data) / original_rate
    new_times = np.linspace(0, duration, num_samples)
    
    return resampled_ecg, new_times

# Save the resampled ECG data as a NumPy .npz file
def save_resampled_ecg_as_npz(subject_number, new_times, resampled_ecg, output_dir="resampled_data"):
    """Save the resampled ECG data in a .npz format (compressed NumPy format)."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subject_number}_resampled_1kHz.npz")

    # Save to NPZ format
    np.savez_compressed(output_path, times=new_times, ecg=resampled_ecg)
    print(f"Resampled data saved to {output_path}")

# Plot the original and resampled data for the first 10 seconds, including a combined plot
def plot_initial_10_seconds(times, ecg_data, new_times, resampled_ecg, subject_number):
    mask_10s_original = times <= 10
    mask_10s_resampled = new_times <= 10

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot original ECG with markers
    ax1.plot(times[mask_10s_original], ecg_data[mask_10s_original], label="Original ECG (128 Hz)", color='blue')
    ax1.scatter(times[mask_10s_original], ecg_data[mask_10s_original], color='blue', s=10)
    ax1.set_title(f'Original ECG Signal (First 10 seconds) - Subject {subject_number}')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax1.legend()

    # Plot resampled ECG with markers
    ax2.plot(new_times[mask_10s_resampled], resampled_ecg[mask_10s_resampled], label="Resampled ECG (1 kHz)", color='orange')
    ax2.scatter(new_times[mask_10s_resampled], resampled_ecg[mask_10s_resampled], color='orange', s=10)
    ax2.set_title(f'Resampled ECG Signal (First 10 seconds) - Subject {subject_number}')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.legend()

    # Combined plot of original and resampled ECG signals with markers
    fig, ax3 = plt.subplots(figsize=(15, 5))
    ax3.plot(times[mask_10s_original], ecg_data[mask_10s_original], label="Original ECG (128 Hz)", color='blue', alpha=0.6)
    ax3.scatter(times[mask_10s_original], ecg_data[mask_10s_original], color='blue', s=10)
    ax3.plot(new_times[mask_10s_resampled], resampled_ecg[mask_10s_resampled], label="Resampled ECG (1 kHz)", color='orange', alpha=0.6)
    ax3.scatter(new_times[mask_10s_resampled], resampled_ecg[mask_10s_resampled], color='orange', s=10)
    ax3.set_title(f'Overlay of Original and Resampled ECG Signal (First 10 seconds) - Subject {subject_number}')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.show()

# Load, resample, and save MIT-BIH ECG data
def load_resample_and_save_ecg(base_path, subject_number):
    record_path = os.path.join(base_path, str(subject_number))
    
    try:
        record = wfdb.rdrecord(record_path)
        ecg_data = record.p_signal[:, 0]
        original_rate = record.fs
        times = np.arange(len(ecg_data)) / original_rate
        
        print(f"Original sampling rate: {original_rate} Hz")
        print(f"Resampling to 1 kHz...")

        resampled_ecg, new_times = resample_ecg_signal(ecg_data, original_rate, 1000)

        # Save resampled data in .npz format
        save_resampled_ecg_as_npz(subject_number, new_times, resampled_ecg)
        
        # Plot original and resampled data for the initial 10 seconds
        plot_initial_10_seconds(times, ecg_data, new_times, resampled_ecg, subject_number)
        
        return resampled_ecg, new_times
    
    except FileNotFoundError:
        print(f"No data found for subject {subject_number}. Please check the subject number.")
        return None, None

# Main function
def main():
    base_path = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2024.11.01\ECG Database\Database\(Healthy- 128 Hz) MIT-BIH Normal Sinus Rhythm Database"
    
    subject_number = input("Enter subject number (e.g., 16265): ")
    resampled_ecg, new_times = load_resample_and_save_ecg(base_path, subject_number)
    
    if resampled_ecg is not None:
        print("Resampling complete, data saved, and plotted.")
    else:
        print("Could not load or resample ECG data.")

if __name__ == "__main__":
    main()
