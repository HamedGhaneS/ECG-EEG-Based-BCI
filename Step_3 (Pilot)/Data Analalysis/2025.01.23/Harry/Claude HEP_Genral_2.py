import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Define groups of electrodes
frontocentral = ['F1', 'F2', 'Fz', 'FC1', 'FC2', 'FCz']
centroparietal = ['C1', 'C2', 'Cz', 'CP1', 'CP2', 'CPz']
electrode_groups = {
    "Frontocentral": frontocentral,
    "Centroparietal": centroparietal
}

# Define all conditions and their event mappings
analysis_conditions = {
    "abs_pe": {
        "folder": "abs_pe_epochs",
        "events": {"high_pe": 1, "low_pe": 2}
    },
    "feedback": {
        "folder": "feedback_epochs",
        "events": {"win": 1, "loss": 2}
    },
    "pe_sign": {
        "folder": "pe_sign_epochs",
        "events": {"positive_pe": 1, "negative_pe": 2}
    }
}

# Updated base path
base_path = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.23\Pilot Data Analysis\Harry\Epoching"

# Plot Saving Path
base_path_out = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.23\Pilot Data Analysis\Harry\HEP Analysis\Plots"

# Function to get epoch files for a condition
def get_epoch_files(condition_folder):
    return {
        'R1': os.path.join(base_path, condition_folder, 'R1_Locked-prepro_epochs-epo.fif'),
        'R2': os.path.join(base_path, condition_folder, 'R2_Locked-prepro_epochs-epo.fif'),
        'R3': os.path.join(base_path, condition_folder, 'R3_Locked-prepro_epochs-epo.fif'),
        'Outcome': os.path.join(base_path, condition_folder, 'Outcome_Locked-prepro_epochs-epo.fif')
    }

def find_continuous_periods(times, sig_times):
    """Find continuous significant periods and their lengths."""
    if len(sig_times) == 0:
        return []
    
    sig_indices = np.where(np.isin(times, sig_times))[0]
    breaks = np.where(np.diff(sig_indices) > 1)[0] + 1
    periods = np.split(sig_indices, breaks)
    
    time_periods = []
    for period in periods:
        if len(period) >= 15:  # Only include periods of at least 15 timepoints (15ms)
            start_time = times[period[0]]
            end_time = times[period[-1]]
            duration = end_time - start_time
            time_periods.append({
                'start': start_time,
                'end': end_time,
                'duration': duration
            })
    
    return time_periods

def process_and_plot(epochs_dict, channels, ax, title, condition_info, is_single_peak=False, peak_type=None):
    """Modified process_and_plot to handle different conditions"""
    event_names = list(condition_info['events'].keys())
    
    if is_single_peak:
        epochs = epochs_dict[peak_type]
    else:
        if peak_type == 'Outcome':
            epochs = epochs_dict['Outcome']
        else:
            # Average across R-peaks
            cond1_list = []
            cond2_list = []
            for peak in ['R1', 'R2', 'R3']:
                epochs = epochs_dict[peak]
                epochs.apply_baseline(baseline=(-0.2, -0.05))
                cond1_list.append(epochs[event_names[0]].copy().pick(channels).get_data().mean(axis=1))
                cond2_list.append(epochs[event_names[1]].copy().pick(channels).get_data().mean(axis=1))
            cond1_data = np.mean(cond1_list, axis=0)
            cond2_data = np.mean(cond2_list, axis=0)
            times = epochs.times
            print(f"\nAveraged R-peaks analysis for {title}")
            return plot_results(cond1_data, cond2_data, times, ax, title, event_names)

    epochs.apply_baseline(baseline=(-0.2, -0.05))
    cond1_data = epochs[event_names[0]].copy().pick(channels).get_data().mean(axis=1)
    cond2_data = epochs[event_names[1]].copy().pick(channels).get_data().mean(axis=1)
    times = epochs.times
    
    print(f"\nAnalysis for {title}")
    return plot_results(cond1_data, cond2_data, times, ax, title, event_names)

def plot_results(cond1_data, cond2_data, times, ax, title, event_names):
    """Modified plot_results to use dynamic condition names"""
    # Calculate statistics
    mean_cond1 = np.mean(cond1_data, axis=0)
    sem_cond1 = stats.sem(cond1_data, axis=0)
    mean_cond2 = np.mean(cond2_data, axis=0)
    sem_cond2 = stats.sem(cond2_data, axis=0)
    
    # Print statistics
    print(f"{event_names[0]} trials: mean range = [{np.min(mean_cond1):.2f}, {np.max(mean_cond1):.2f}] μV")
    print(f"{event_names[1]} trials: mean range = [{np.min(mean_cond2):.2f}, {np.max(mean_cond2):.2f}] μV")
    print(f"Number of trials - {event_names[0]}: {len(cond1_data)}, {event_names[1]}: {len(cond2_data)}")
    
    # Calculate significance
    t_stats, p_values = stats.ttest_ind(cond1_data, cond2_data, axis=0)
    sig_times = times[p_values < 0.05]
    sig_periods = find_continuous_periods(times, sig_times)
    
    # Create plot
    ax.plot(times, mean_cond1, 'r-', label=event_names[0])
    ax.plot(times, mean_cond2, 'b-', label=event_names[1])
    
    # Add SEM shading
    ax.fill_between(times, mean_cond1-sem_cond1, mean_cond1+sem_cond1, color='red', alpha=0.2)
    ax.fill_between(times, mean_cond2-sem_cond2, mean_cond2+sem_cond2, color='blue', alpha=0.2)
    
    # Add significance markers
    if sig_periods:
        ylims = ax.get_ylim()
        yrange = ylims[1] - ylims[0]
        text_y = ylims[1] + yrange * 0.05
        
        for period in sig_periods:
            ax.axvspan(period['start'], period['end'], color='grey', alpha=0.3, zorder=0)
            mid_x = (period['start'] + period['end']) / 2
            ax.text(mid_x, text_y, f'{period["duration"]*1000:.0f}ms', 
                   horizontalalignment='center', verticalalignment='bottom',
                   fontsize=8)
    
    # Style the plot
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Amplitude (μV)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.2, 0.6)
    
    if sig_periods:
        ax.set_ylim(ylims[0], text_y + yrange * 0.1)

def main():
    # Set figure style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'lines.linewidth': 1.5
    })
    
    # Process each condition separately
    for condition_name, condition_info in analysis_conditions.items():
        print(f"\nProcessing {condition_name} condition...")
        
        # Get epoch files for this condition
        epoch_files = get_epoch_files(condition_info['folder'])
        
        # Load epoch files
        epoch_data = {}
        for epoch_type, file_path in epoch_files.items():
            print(f"Loading {epoch_type} epochs from {file_path}")
            epochs = mne.read_epochs(file_path, preload=True)
            epoch_data[epoch_type] = epochs
        
        # Create figures for this condition
        fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
        fig1.suptitle(f'Combined HEP Analysis - {condition_name.upper()}', fontsize=12)
        
        # Plot combined analyses
        for i, (group_name, channels) in enumerate(electrode_groups.items()):
            process_and_plot(epoch_data, channels, axes1[i, 0], 
                           f'Outcome-locked HEP: {group_name}',
                           condition_info, peak_type='Outcome')
            process_and_plot(epoch_data, channels, axes1[i, 1],
                           f'Averaged R-peaks HEP: {group_name}',
                           condition_info)
        
        fig1.tight_layout()
        
        # Create figure for individual R-peaks
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
        fig2.suptitle(f'Individual R-peak HEP Analysis - {condition_name.upper()}', fontsize=12)
        
        # Plot individual R-peak analyses
        for i, (group_name, channels) in enumerate(electrode_groups.items()):
            for j, peak in enumerate(['R1', 'R2', 'R3']):
                process_and_plot(epoch_data, channels, axes2[i, j],
                               f'{peak}-locked HEP: {group_name}',
                               condition_info, is_single_peak=True, peak_type=peak)
        
        fig2.tight_layout()
        
        # Save figures
        fig1.savefig(os.path.join(base_path_out, f'{condition_name}_combined_hep_analysis.png'), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(base_path_out, f'{condition_name}_individual_peaks_analysis.png'), dpi=300, bbox_inches='tight')
        
    plt.show()

if __name__ == "__main__":
    main()