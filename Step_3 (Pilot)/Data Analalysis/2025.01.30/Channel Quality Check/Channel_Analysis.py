import pyxdf
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal
from eeg_montage import CHANNEL_NAMES
import os

def get_participant_info():
    """Return dictionary of participant-specific information"""
    return {
        "Elmira": {
            "xdf_path": r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.13\Pilot\Elmira\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf",
            "stream_index": 3
        },
        "Harry": {
            "xdf_path": r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf",
            "stream_index": 2
        }
    }

def calculate_channel_metrics(data, srate):
    """Calculate metrics for channel quality assessment"""
    # Variance
    variance = np.var(data)
    
    # Kurtosis
    kurtosis = stats.kurtosis(data)
    
    # Line noise ratio (around 50Hz)
    freqs, psd = signal.welch(data, srate, nperseg=int(srate))
    line_noise_idx = np.argmin(np.abs(freqs - 50))
    line_noise_power = np.mean(psd[line_noise_idx-1:line_noise_idx+2])
    total_power = np.mean(psd)
    line_noise_ratio = line_noise_power / total_power
    
    # High frequency ratio (above 40Hz)
    hf_idx = np.where(freqs > 40)[0]
    hf_power = np.mean(psd[hf_idx])
    hf_ratio = hf_power / total_power
    
    return {
        'variance': variance,
        'kurtosis': kurtosis,
        'line_noise_ratio': line_noise_ratio,
        'hf_ratio': hf_ratio
    }

def plot_metrics(metrics_dict, channel_names, participant):
    """Plot channel quality metrics"""
    metrics = ['variance', 'kurtosis', 'line_noise_ratio', 'hf_ratio']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 4*len(metrics)))
    fig.suptitle(f'Channel Quality Metrics - {participant}')
    
    for idx, metric in enumerate(metrics):
        values = [metrics_dict[ch][metric] for ch in channel_names]
        z_scores = stats.zscore(values)
        outliers = np.abs(z_scores) > 2
        
        axes[idx].bar(range(len(channel_names)), values, 
                     color=['red' if outlier else 'blue' for outlier in outliers])
        axes[idx].set_xticks(range(len(channel_names)))
        axes[idx].set_xticklabels(channel_names, rotation=45, ha='right')
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Distribution')
        axes[idx].grid(True, alpha=0.3)
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        axes[idx].axhline(y=mean_val + 2*std_val, color='r', linestyle='--', alpha=0.5)
        axes[idx].axhline(y=mean_val - 2*std_val, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

def plot_raw_channels(eeg_data, srate, participant, save_dir):
    """Plot raw EEG channels in groups"""
    duration = 10  # seconds
    time = np.arange(int(srate * duration)) / srate
    channels_per_fig = 16
    n_figures = int(np.ceil(64 / channels_per_fig))
    
    for fig_num in range(n_figures):
        start_ch = fig_num * channels_per_fig
        end_ch = min(start_ch + channels_per_fig, 64)
        
        fig = plt.figure(figsize=(15, 20))
        plt.suptitle(f"{participant} - Raw EEG Channels {start_ch+1} to {end_ch}")
        
        for i in range(start_ch, end_ch):
            ax = plt.subplot(channels_per_fig, 1, i - start_ch + 1)
            
            # Get data for this channel
            channel_data = eeg_data[i, :int(srate * duration)]
            
            # Calculate y-limits
            data_range = np.ptp(channel_data)
            data_mean = np.mean(channel_data)
            y_margin = data_range * 0.1
            ylim_min = data_mean - (data_range/2 + y_margin)
            ylim_max = data_mean + (data_range/2 + y_margin)
            
            # Plot
            plt.plot(time, channel_data, 'k-', linewidth=0.5)
            plt.ylabel(f'{CHANNEL_NAMES[i]}\n(μV)')
            plt.grid(True, alpha=0.3)
            plt.ylim(ylim_min, ylim_max)
            
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.1)
            
            if i < end_ch - 1:
                plt.xticks([])
            else:
                plt.xlabel('Time (s)')
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(save_dir, f'{participant}_raw_channels_{start_ch+1}_to_{end_ch}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

def analyze_participant(participant, output_dir):
    """Analyze EEG data for a single participant"""
    print(f"\nProcessing {participant}'s data...")
    
    # Create participant directory
    participant_dir = os.path.join(output_dir, participant)
    os.makedirs(participant_dir, exist_ok=True)
    
    # Get participant info
    participant_info = get_participant_info()[participant]
    
    # Load XDF file
    print(f"Loading XDF file...")
    streams, _ = pyxdf.load_xdf(participant_info["xdf_path"])
    
    # Get EEG stream
    eeg_stream = streams[participant_info["stream_index"]]
    eeg_data = np.array(eeg_stream['time_series']).T
    srate = float(eeg_stream['info']['nominal_srate'][0])
    
    # Plot raw channels
    print("Generating raw channel plots...")
    plot_raw_channels(eeg_data, srate, participant, participant_dir)
    
    # Calculate metrics
    print("Calculating channel metrics...")
    metrics_dict = {}
    for i, channel in enumerate(CHANNEL_NAMES[:-1]):
        metrics_dict[channel] = calculate_channel_metrics(eeg_data[i], srate)
    
    # Plot metrics
    fig = plot_metrics(metrics_dict, CHANNEL_NAMES[:-1], participant)
    plot_path = os.path.join(participant_dir, f'{participant}_channel_metrics.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Generate report
    print("Generating analysis report...")
    report_path = os.path.join(participant_dir, f'{participant}_channel_analysis.txt')
    with open(report_path, 'w') as f:
        f.write(f"Channel Quality Analysis Report for {participant}\n")
        f.write("="*50 + "\n\n")
        
        for metric in ['variance', 'kurtosis', 'line_noise_ratio', 'hf_ratio']:
            f.write(f"\n{metric.replace('_', ' ').title()}:\n")
            f.write("-"*30 + "\n")
            
            values = [metrics_dict[ch][metric] for ch in CHANNEL_NAMES[:-1]]
            z_scores = stats.zscore(values)
            
            for ch_idx, z_score in enumerate(z_scores):
                if abs(z_score) > 2:
                    f.write(f"{CHANNEL_NAMES[ch_idx]}: z-score = {z_score:.2f}\n")
        
        # Summary
        f.write("\nSummary of Problematic Channels:\n")
        f.write("-"*30 + "\n")
        
        bad_channels = set()
        for metric in ['variance', 'kurtosis', 'line_noise_ratio', 'hf_ratio']:
            values = [metrics_dict[ch][metric] for ch in CHANNEL_NAMES[:-1]]
            z_scores = stats.zscore(values)
            for ch_idx, z_score in enumerate(z_scores):
                if abs(z_score) > 2:
                    bad_channels.add(CHANNEL_NAMES[ch_idx])
        
        f.write("Channels with metrics exceeding 2 standard deviations:\n")
        for channel in sorted(bad_channels):
            f.write(f"- {channel}\n")
    
    print(f"Analysis completed for {participant}")
    print(f"Results saved in: {participant_dir}")

def main():
    # Set output directory
    output_dir = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.30\Bad Channel Removal"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each participant
    participants = ["Elmira", "Harry"]
    for participant in participants:
        try:
            analyze_participant(participant, output_dir)
        except Exception as e:
            print(f"Error processing {participant}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
