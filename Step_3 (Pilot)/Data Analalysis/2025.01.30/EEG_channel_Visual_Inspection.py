import mne
import matplotlib.pyplot as plt
import os
import numpy as np
import pyxdf
from eeg_montage import CHANNEL_NAMES

def get_participant_paths(participant):
    """Get the appropriate file paths and stream index for the participant"""
    if participant == "Elmira":
        return {
            'xdf_file': r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.13\Pilot\Elmira\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf",
            'stream_index': 3
        }
    elif participant == "Harry":
        return {
            'xdf_file': r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf",
            'stream_index': 2
        }
    else:
        raise ValueError(f"Unknown participant: {participant}")

def plot_eeg_channels(raw, output_dir, participant_name):
    """Plot EEG channels in groups of 8 for visual inspection"""
    
    # Get all channel names
    all_channels = raw.ch_names
    
    # Calculate number of channels and figures needed
    n_channels = len(all_channels)
    channels_per_plot = 8
    n_figures = int(np.ceil(n_channels / channels_per_plot))
    
    # Set up plotting parameters
    duration = 10  # Show 10 seconds of data in seconds
    start_time = 0  # Start from the beginning
    samples_to_plot = int(duration * raw.info['sfreq'])
    
    # Create plots
    for fig_num in range(n_figures):
        # Calculate channel indices for this figure
        start_idx = fig_num * channels_per_plot
        end_idx = min(start_idx + channels_per_plot, n_channels)
        channels_to_plot = all_channels[start_idx:end_idx]
        
        # Create figure
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle(f'{participant_name} - Channels {start_idx+1} to {end_idx}', fontsize=12)
        
        # Get data for these channels
        data, times = raw[:len(channels_to_plot), 
                         int(start_time * raw.info['sfreq']):
                         int((start_time + duration) * raw.info['sfreq'])]
        
        # Plot each channel in this group
        for i, (channel, channel_data) in enumerate(zip(channels_to_plot, data)):
            ax = plt.subplot(channels_per_plot, 1, i+1)
            
            # Plot the data
            ax.plot(times, channel_data * 1e6, 'k-', linewidth=0.5)  # Convert to μV
            
            # Style the plot
            ax.set_ylabel(f'{channel}\n(μV)', rotation=0, ha='right', va='center')
            ax.grid(True, alpha=0.3)
            if i < len(channels_to_plot) - 1:
                ax.set_xticks([])
            
            # Set y-axis limits to ±75 μV for better visualization
            ax.set_ylim(-75, 75)
            
            # Only show x-axis label on the bottom subplot
            if i == len(channels_to_plot) - 1:
                ax.set_xlabel('Time (s)')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(output_dir, f'{participant_name}_channels_{start_idx+1}_to_{end_idx}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved figure {fig_num+1}/{n_figures} to {fig_path}")

def main():
    while True:
        # Ask for participant name
        participant = input("Enter participant name (Elmira/Harry) or 'q' to quit: ").strip()
        
        if participant.lower() == 'q':
            print("Exiting program...")
            break
            
        if participant not in ["Elmira", "Harry"]:
            print(f"Unknown participant: {participant}. Please try again.")
            continue
            
        try:
            # Get participant paths and stream index
            participant_info = get_participant_paths(participant)
            print(f"\nProcessing data for participant: {participant}")
            
            # Create output directory
            output_dir = os.path.join(os.getcwd(), 'Channel_Inspection', participant)
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory: {output_dir}")
            
            # Load XDF file
            print(f"Loading XDF file from: {participant_info['xdf_file']}")
            streams, header = pyxdf.load_xdf(participant_info['xdf_file'])
            
            # Get EEG stream using correct index
            eeg_stream = streams[participant_info['stream_index']]
            print(f"Using EEG stream index: {participant_info['stream_index']}")
            
            # Extract EEG data and sampling rate
            eeg_data = np.array(eeg_stream['time_series']).T
            srate = float(eeg_stream['info']['nominal_srate'][0])
            
            print(f"Data shape: {eeg_data.shape}")
            print(f"Sampling rate: {srate} Hz")
            
            # Create MNE raw object
            info = mne.create_info(CHANNEL_NAMES, srate, ch_types='eeg')
            raw = mne.io.RawArray(eeg_data, info)
            
            # Plot channels
            plot_eeg_channels(raw, output_dir, participant)
            
            print(f"\nCompleted channel visualization for {participant}")
            
        except Exception as e:
            print(f"Error processing {participant}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()