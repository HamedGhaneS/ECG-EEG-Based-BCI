import pyxdf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load data
data, header = pyxdf.load_xdf(str(r'H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.13\Pilot\Elmira\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf'))
rpeak_df = pd.read_csv(r'H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.13\Pilot\Elmira\Elmira-ses001-run1-20250113-173456_timing_pairs.CSV')

ecg_stream = data[-1]
ecg_data = ecg_stream['time_series'][:, -1]
ecg_timestamps = ecg_stream['time_stamps']

mean_offset = np.mean(rpeak_df['calculated_offset'])
print(f"Mean PC1-PC2 offset: {mean_offset:.3f}s")

# Shift timestamps in opposite direction of the offset
shifted_timestamps = ecg_timestamps - mean_offset  # Changed from + to -

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

# Original ECG plot
ax1.plot(ecg_timestamps, ecg_data, 'b-', linewidth=0.5, label='Original ECG')
for peak_time in rpeak_df['pc1_lsl_time']:
    idx = np.searchsorted(ecg_timestamps, peak_time)
    if idx < len(ecg_data):
        ax1.plot(peak_time, ecg_data[idx], 'ro', markersize=6)
for peak_time in rpeak_df['pc2_lsl_time']:
    idx = np.searchsorted(ecg_timestamps, peak_time)
    if idx < len(ecg_data):
        ax1.plot(peak_time, ecg_data[idx], 'go', markersize=6)

ax1.grid(True)
ax1.set_ylabel('ECG Amplitude')
ax1.set_title('Original ECG with R-peaks (Red: PC1, Green: PC2)')
ax1.plot([], [], 'ro', label='PC1 R-peaks')
ax1.plot([], [], 'go', label='PC2 R-peaks')
ax1.legend()

# Shifted ECG plot
ax2.plot(shifted_timestamps, ecg_data, 'b-', linewidth=0.5, label='Shifted ECG')
for peak_time in rpeak_df['pc1_lsl_time']:
    idx = np.searchsorted(ecg_timestamps, peak_time)
    if idx < len(ecg_data):
        ax2.plot(peak_time, ecg_data[idx], 'ro', markersize=6)
for peak_time in rpeak_df['pc2_lsl_time']:
    idx = np.searchsorted(ecg_timestamps, peak_time)
    if idx < len(ecg_data):
        ax2.plot(peak_time, ecg_data[idx], 'go', markersize=6)

ax2.grid(True)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('ECG Amplitude')
ax2.set_title(f'Shifted ECG (by {-mean_offset:.3f}s) with R-peaks')
ax2.plot([], [], 'ro', label='PC1 R-peaks')
ax2.plot([], [], 'go', label='PC2 R-peaks')
ax2.legend()

plt.tight_layout()
plt.show()

print("\nTiming Statistics:")
print(f"Recording duration: {ecg_timestamps[-1] - ecg_timestamps[0]:.1f}s")
print(f"Number of R-peaks: {len(rpeak_df)}")
print(f"Std of offset: {np.std(rpeak_df['calculated_offset']):.6f}s")