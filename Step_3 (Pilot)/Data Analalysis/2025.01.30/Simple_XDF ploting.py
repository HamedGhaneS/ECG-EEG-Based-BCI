import pyxdf
import matplotlib.pyplot as plt
import numpy as np
from eeg_montage import CHANNEL_NAMES

# Load the XDF file
file_path = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.13\Pilot\Elmira\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
streams, header = pyxdf.load_xdf(file_path)

# Get Elmira's EEG stream (index 3)
eeg_stream = streams[3]
eeg_data = np.array(eeg_stream['time_series']).T
srate = float(eeg_stream['info']['nominal_srate'][0])

# Create time vector for first 10 seconds
duration = 10  # seconds
time = np.arange(int(srate * duration)) / srate

# Number of channels to plot (let's do 16 channels)
n_channels = 16
channels_per_fig = 16

# Calculate how many figures we need
n_figures = int(np.ceil(64 / channels_per_fig))

# Plot figures
for fig_num in range(n_figures):
    start_ch = fig_num * channels_per_fig
    end_ch = min(start_ch + channels_per_fig, 64)
    
    plt.figure(figsize=(15, 20))
    plt.suptitle(f"Elmira - Raw EEG Channels {start_ch+1} to {end_ch}")
    
    for i in range(start_ch, end_ch):
        ax = plt.subplot(channels_per_fig, 1, i - start_ch + 1)
        
        # Get data for this channel
        channel_data = eeg_data[i, :int(srate * duration)]
        
        # Calculate good y-limits for this channel
        data_range = np.ptp(channel_data)
        data_mean = np.mean(channel_data)
        y_margin = data_range * 0.1  # Add 10% margin
        ylim_min = data_mean - (data_range/2 + y_margin)
        ylim_max = data_mean + (data_range/2 + y_margin)
        
        # Plot the data
        plt.plot(time, channel_data, 'k-', linewidth=0.5)
        plt.ylabel(f'{CHANNEL_NAMES[i]}\n(μV)')
        plt.grid(True, alpha=0.3)
        plt.ylim(ylim_min, ylim_max)
        
        # Add a zero line
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.1)
        
        # Only show x-axis for bottom plot
        if i < end_ch - 1:
            plt.xticks([])
        else:
            plt.xlabel('Time (s)')
            
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

    # Ask if user wants to continue to next set of channels
    if fig_num < n_figures - 1:
        resp = input(f"Press Enter to see next {channels_per_fig} channels (or 'q' to quit): ")
        if resp.lower() == 'q':
            break