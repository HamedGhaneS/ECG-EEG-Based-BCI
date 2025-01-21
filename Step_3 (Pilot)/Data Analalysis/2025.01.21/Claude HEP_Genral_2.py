import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Define groups of electrodes
frontocentral = ['F1', 'F2', 'Fz', 'FC1', 'FC2', 'FCz']
centroparietal = ['C1', 'C2', 'Cz', 'CP1', 'CP2', 'CPz']
electrode_groups = {
    "Frontocentral": frontocentral,
    "Centroparietal": centroparietal
}

# Define conditions
conditions = {"high_pe": 1, "low_pe": 2}  # Event IDs

# File paths to preprocessed epochs
base_path = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.21\Pilot Data Analysis"
epoch_files = {
    'R1': base_path + r'\R1_Locked-prepro_hep_epochs-epo.fif',
    'R2': base_path + r'\R2_Locked-prepro_hep_epochs-epo.fif',
    'R3': base_path + r'\R3_Locked-prepro_hep_epochs-epo.fif',
    'Outcome': base_path + r'\Outcome_Locked-prepro_hep_epochs-epo.fif'
}

def find_continuous_periods(times, sig_times):
    """Find continuous significant periods and their lengths."""
    if len(sig_times) == 0:
        return []
    
    # Convert to indices for easier processing
    sig_indices = np.where(np.isin(times, sig_times))[0]
    
    # Find breaks in consecutive indices
    breaks = np.where(np.diff(sig_indices) > 1)[0] + 1
    
    # Split into continuous periods
    periods = np.split(sig_indices, breaks)
    
    # Convert to time ranges
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

def process_and_plot(epochs_dict, channels, ax, title, is_single_peak=False, peak_type=None):
    """Process data and create plot for a specific channel group and peak type"""
    if is_single_peak:
        epochs = epochs_dict[peak_type]
        # Apply baseline correction using pre-stimulus period (-0.2 to -0.05s)
        epochs.apply_baseline(baseline=(-0.2, -0.05))
        high_pe = epochs["high_pe"].copy().pick(channels).get_data().mean(axis=1)
        low_pe = epochs["low_pe"].copy().pick(channels).get_data().mean(axis=1)
    else:
        if peak_type == 'Outcome':
            epochs = epochs_dict['Outcome']
            epochs.apply_baseline(baseline=(-0.2, -0.05))
            high_pe = epochs["high_pe"].copy().pick(channels).get_data().mean(axis=1)
            low_pe = epochs["low_pe"].copy().pick(channels).get_data().mean(axis=1)
        else:
            high_pe_list = []
            low_pe_list = []
            for peak in ['R1', 'R2', 'R3']:
                epochs = epochs_dict[peak]
                epochs.apply_baseline(baseline=(-0.2, -0.05))
                high_pe_list.append(epochs["high_pe"].copy().pick(channels).get_data().mean(axis=1))
                low_pe_list.append(epochs["low_pe"].copy().pick(channels).get_data().mean(axis=1))
            high_pe = np.mean(high_pe_list, axis=0)
            low_pe = np.mean(low_pe_list, axis=0)

    # Get time points
    times = epochs.times
    
    # Calculate means and SEMs
    mean_high = np.mean(high_pe, axis=0)
    sem_high = stats.sem(high_pe, axis=0)
    mean_low = np.mean(low_pe, axis=0)
    sem_low = stats.sem(low_pe, axis=0)
    
    # Print some statistics about the data
    print(f"\nStatistics for {title}:")
    print(f"High PE trials: mean range = [{np.min(mean_high):.2f}, {np.max(mean_high):.2f}] μV")
    print(f"Low PE trials: mean range = [{np.min(mean_low):.2f}, {np.max(mean_low):.2f}] μV")
    print(f"High PE SEM range = [{np.min(sem_high):.2f}, {np.max(sem_high):.2f}] μV")
    print(f"Low PE SEM range = [{np.min(sem_low):.2f}, {np.max(sem_low):.2f}] μV")
    print(f"Number of trials - High PE: {high_pe.shape[0]}, Low PE: {low_pe.shape[0]}")
    
    # Simple t-test at each timepoint
    t_stats, p_values = stats.ttest_ind(high_pe, low_pe, axis=0)
    sig_times = times[p_values < 0.05]
    if is_single_peak:
        epochs = epochs_dict[peak_type]
        high_pe = epochs["high_pe"].copy().pick(channels).get_data().mean(axis=1)
        low_pe = epochs["low_pe"].copy().pick(channels).get_data().mean(axis=1)
    else:
        # Average across R-peaks or use outcome epochs
        if peak_type == 'Outcome':
            epochs = epochs_dict['Outcome']
            high_pe = epochs["high_pe"].copy().pick(channels).get_data().mean(axis=1)
            low_pe = epochs["low_pe"].copy().pick(channels).get_data().mean(axis=1)
        else:  # Average R-peaks
            high_pe_list = []
            low_pe_list = []
            for peak in ['R1', 'R2', 'R3']:
                epochs = epochs_dict[peak]
                high_pe_list.append(epochs["high_pe"].copy().pick(channels).get_data().mean(axis=1))
                low_pe_list.append(epochs["low_pe"].copy().pick(channels).get_data().mean(axis=1))
            high_pe = np.mean(high_pe_list, axis=0)
            low_pe = np.mean(low_pe_list, axis=0)

    # Get time points
    times = epochs.times
    
    # Calculate means and SEMs
    mean_high = np.mean(high_pe, axis=0)
    sem_high = stats.sem(high_pe, axis=0)
    mean_low = np.mean(low_pe, axis=0)
    sem_low = stats.sem(low_pe, axis=0)
    
    # Simple t-test at each timepoint
    t_stats, p_values = stats.ttest_ind(high_pe, low_pe, axis=0)
    sig_times = times[p_values < 0.05]  # Uncorrected p < 0.05
    
    # Find continuous significant periods
    sig_periods = find_continuous_periods(times, sig_times)
    
    # Create plot
    ax.plot(times, mean_high, 'r-', label='High PE')
    ax.plot(times, mean_low, 'b-', label='Low PE')
    
    # Add SEM shading
    ax.fill_between(times, mean_high-sem_high, mean_high+sem_high, color='red', alpha=0.2)
    ax.fill_between(times, mean_low-sem_low, mean_low+sem_low, color='blue', alpha=0.2)
    
    # Add significance markers and annotations
    if sig_periods:
        ylims = ax.get_ylim()
        yrange = ylims[1] - ylims[0]
        text_y = ylims[1] + yrange * 0.05  # Place text above the plot
        
        for i, period in enumerate(sig_periods):
            # Add shading
            ax.axvspan(period['start'], period['end'], color='grey', alpha=0.3, zorder=0)
            
            # Add duration text
            mid_x = (period['start'] + period['end']) / 2
            ax.text(mid_x, text_y, f'{period["duration"]*1000:.0f}ms', 
                   horizontalalignment='center', verticalalignment='bottom',
                   fontsize=8)
    
    # Add statistical test information to legend
    ax.plot([], [], ' ', label=f'Statistics: Unpaired t-test\np < 0.05 uncorrected')
    
    # Style the plot
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Amplitude (μV)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set x-axis limits to match paper
    ax.set_xlim(-0.2, 0.6)
    
    # Adjust y-limits to accommodate duration text
    if sig_periods:
        ax.set_ylim(ylims[0], text_y + yrange * 0.1)
    """Process data and create plot for a specific channel group and peak type"""
    if is_single_peak:
        epochs = epochs_dict[peak_type]
        high_pe = epochs["high_pe"].copy().pick(channels).get_data().mean(axis=1)
        low_pe = epochs["low_pe"].copy().pick(channels).get_data().mean(axis=1)
    else:
        # Average across R-peaks or use outcome epochs
        if peak_type == 'Outcome':
            epochs = epochs_dict['Outcome']
            high_pe = epochs["high_pe"].copy().pick(channels).get_data().mean(axis=1)
            low_pe = epochs["low_pe"].copy().pick(channels).get_data().mean(axis=1)
        else:  # Average R-peaks
            high_pe_list = []
            low_pe_list = []
            for peak in ['R1', 'R2', 'R3']:
                epochs = epochs_dict[peak]
                high_pe_list.append(epochs["high_pe"].copy().pick(channels).get_data().mean(axis=1))
                low_pe_list.append(epochs["low_pe"].copy().pick(channels).get_data().mean(axis=1))
            high_pe = np.mean(high_pe_list, axis=0)
            low_pe = np.mean(low_pe_list, axis=0)

    # Get time points
    times = epochs.times
    
    # Calculate means and SEMs
    mean_high = np.mean(high_pe, axis=0)
    sem_high = stats.sem(high_pe, axis=0)
    mean_low = np.mean(low_pe, axis=0)
    sem_low = stats.sem(low_pe, axis=0)
    
    # Simple t-test at each timepoint
    t_stats, p_values = stats.ttest_ind(high_pe, low_pe, axis=0)
    sig_times = times[p_values < 0.05]  # Uncorrected p < 0.05
    
    # Create plot
    ax.plot(times, mean_high, 'r-', label='High PE')
    ax.plot(times, mean_low, 'b-', label='Low PE')
    
    # Add SEM shading
    ax.fill_between(times, mean_high-sem_high, mean_high+sem_high, color='red', alpha=0.2)
    ax.fill_between(times, mean_low-sem_low, mean_low+sem_low, color='blue', alpha=0.2)
    
    # Add significance markers
    if len(sig_times) > 0:
        ylims = ax.get_ylim()
        for t in sig_times:
            ax.axvspan(t-0.001, t+0.001, color='grey', alpha=0.3, zorder=0)
    
    # Style the plot
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Amplitude (μV)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set x-axis limits to match paper
    ax.set_xlim(-0.2, 0.6)

def main():
    # Set figure style
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
    
    # Load all preprocessed epoch files
    epoch_data = {}
    for epoch_type, file_path in epoch_files.items():
        print(f"\nProcessing {epoch_type} epochs...")
        epochs = mne.read_epochs(file_path, preload=True)
        epoch_data[epoch_type] = epochs

    # Create figure for combined plots (2x2)
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
    fig1.suptitle('Combined HEP Analysis', fontsize=12)

    # Plot combined analyses
    for i, (group_name, channels) in enumerate(electrode_groups.items()):
        # Outcome-locked
        process_and_plot(epoch_data, channels, axes1[i, 0], 
                        f'Outcome-locked HEP: {group_name}',
                        peak_type='Outcome')
        # Averaged R-peaks
        process_and_plot(epoch_data, channels, axes1[i, 1],
                        f'Averaged R-peaks HEP: {group_name}')

    plt.tight_layout()

    # Create figure for individual R-peak plots (2x3)
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Individual R-peak HEP Analysis', fontsize=12)

    # Plot individual R-peak analyses
    for i, (group_name, channels) in enumerate(electrode_groups.items()):
        for j, peak in enumerate(['R1', 'R2', 'R3']):
            process_and_plot(epoch_data, channels, axes2[i, j],
                           f'{peak}-locked HEP: {group_name}',
                           is_single_peak=True, peak_type=peak)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()