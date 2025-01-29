import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Define groups of electrodes using channel names
electrode_groups = {
    # HEP analysis groups
    "Frontocentral": ['F1', 'F2', 'Fz', 'FC1', 'FC2', 'FCz'],
    "Centroparietal": ['C1', 'C2', 'Cz', 'CP1', 'CP2', 'CPz'],
    # VEP analysis groups
    "Primary_VEP": ['O1', 'O2', 'Oz'],  # Primary visual cortex
    "Extended_VEP": ['PO7', 'PO8', 'PO3', 'PO4', 'P7', 'P8', 'P3', 'P4', 'Pz']  # Visual association areas
}

# Create electrode legend labels
electrode_legend = {
    "Primary_VEP": "VEP (O1, O2, Oz)",
    "Extended_VEP": "VEP (PO7, PO8, PO3, PO4, P7, P8, P3, P4, Pz)"
}

# Separate groups for different analyses
HEP_GROUPS = ["Frontocentral", "Centroparietal"]
VEP_GROUPS = ["Primary_VEP", "Extended_VEP"]

# Define conditions and their event mappings
analysis_conditions = {
    "abs_pe": {
        "folder": "abs_pe_epochs",
        "events": {"high_pe": 1, "low_pe": 2}
    },
    "pe_sign": {
        "folder": "pe_sign_epochs",
        "events": {"positive_pe": 1, "negative_pe": 2}
    },
    "vep": {
        "folder": "vep_epochs",
        "events": {"all": 1}
    }
}

def find_continuous_periods(times, sig_times):
    """Find continuous significant periods and their lengths after t=0."""
    if len(sig_times) == 0:
        return []
    
    # Only include times after t=0
    sig_times = sig_times[sig_times >= 0]
    
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
    """Process and plot HEP data"""
    event_names = list(condition_info['events'].keys())
    
    if is_single_peak:
        epochs = epochs_dict[peak_type].copy()
    else:
        if peak_type == 'Outcome':
            epochs = epochs_dict['Outcome'].copy()
        else:
            # Average across R-peaks
            cond1_list = []
            cond2_list = []
            for peak in ['R1', 'R2', 'R3']:
                peak_epochs = epochs_dict[peak].copy()
                peak_epochs.pick(channels)
                peak_epochs.apply_baseline(baseline=(-0.2, -0.05))
                
                cond1_list.append(peak_epochs[event_names[0]].get_data().mean(axis=1))
                cond2_list.append(peak_epochs[event_names[1]].get_data().mean(axis=1))
            
            cond1_data = np.mean(cond1_list, axis=0)
            cond2_data = np.mean(cond2_list, axis=0)
            times = peak_epochs.times
            print(f"\nAveraged R-peaks analysis for {title}")
            return plot_results(cond1_data, cond2_data, times, ax, title, event_names)

    epochs.pick(channels)
    epochs.apply_baseline(baseline=(-0.2, -0.05))
    
    cond1_data = epochs[event_names[0]].get_data().mean(axis=1)
    cond2_data = epochs[event_names[1]].get_data().mean(axis=1)
    times = epochs.times
    
    print(f"\nAnalysis for {title}")
    return plot_results(cond1_data, cond2_data, times, ax, title, event_names)

def process_and_plot(epochs_dict, channels, ax, title, condition_info, 
                     is_single_peak=False, peak_type=None, invert=False):
    """Process and plot HEP data with optional polarity inversion"""
    event_names = list(condition_info['events'].keys())

    if is_single_peak:
        epochs = epochs_dict[peak_type].copy()
    else:
        if peak_type == 'Outcome':
            epochs = epochs_dict['Outcome'].copy()
        else:
            # Average across R-peaks
            cond1_list = []
            cond2_list = []
            for peak in ['R1', 'R2', 'R3']:
                peak_epochs = epochs_dict[peak].copy()
                peak_epochs.pick(channels)
                peak_epochs.apply_baseline(baseline=(-0.2, -0.05))

                cond1_list.append(peak_epochs[event_names[0]].get_data().mean(axis=1))
                cond2_list.append(peak_epochs[event_names[1]].get_data().mean(axis=1))

            cond1_data = np.mean(cond1_list, axis=0)
            cond2_data = np.mean(cond2_list, axis=0)
            times = peak_epochs.times
            print(f"\nAveraged R-peaks analysis for {title} (Inverted: {invert})")

            # Apply polarity inversion if needed
            if invert:
                cond1_data = -cond1_data
                cond2_data = -cond2_data
                title = "Inverted " + title

            return plot_results(cond1_data, cond2_data, times, ax, title, event_names)

    epochs.pick(channels)
    epochs.apply_baseline(baseline=(-0.2, -0.05))

    cond1_data = epochs[event_names[0]].get_data().mean(axis=1)
    cond2_data = epochs[event_names[1]].get_data().mean(axis=1)
    times = epochs.times

    print(f"\nAnalysis for {title} (Inverted: {invert})")

    # Apply polarity inversion if needed
    if invert:
        cond1_data = -cond1_data
        cond2_data = -cond2_data
        title = "Inverted " + title

    return plot_results(cond1_data, cond2_data, times, ax, title, event_names)


def plot_results(cond1_data, cond2_data, times, ax, title, event_names):
    """Plot HEP results with statistics"""
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

def get_participant_paths(participant):
    """Get the appropriate file paths for the participant"""
    if participant == "Elmira":
        return r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.28\Pilot Data Analysis\Enhanced Epoching\Elmira"
    elif participant == "Harry":
        return r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.28\Pilot Data Analysis\Enhanced Epoching\Harry"
    else:
        raise ValueError(f"Unknown participant: {participant}")

def get_epoch_files(base_path, condition_folder):
    """Get epoch files for a condition"""
    if condition_folder == "vep_epochs":
        return {
            'Stim': os.path.join(base_path, condition_folder, 'Stim_Locked-prepro_epochs-epo.fif'),
            'Outcome': os.path.join(base_path, condition_folder, 'Outcome_All_Locked-prepro_epochs-epo.fif')
        }
    else:
        return {
            'R1': os.path.join(base_path, condition_folder, 'R1_Locked-prepro_epochs-epo.fif'),
            'R2': os.path.join(base_path, condition_folder, 'R2_Locked-prepro_epochs-epo.fif'),
            'R3': os.path.join(base_path, condition_folder, 'R3_Locked-prepro_epochs-epo.fif'),
            'Outcome': os.path.join(base_path, condition_folder, 'Outcome_Locked-prepro_epochs-epo.fif')
        }

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
            # Get base path for the participant
            base_path = get_participant_paths(participant)
            print(f"\nProcessing data for participant: {participant}")
            print(f"Base path: {base_path}")
            
            # Check if base path exists
            if not os.path.exists(base_path):
                print(f"Warning: Base path does not exist: {base_path}")
                continue
            
            # Create output directory
            output_dir = os.path.join(base_path, 'Analysis_Plots')
            os.makedirs(output_dir, exist_ok=True)
            
            # Process each condition
            for condition_name, condition_info in analysis_conditions.items():
                print(f"\nProcessing {condition_name} condition...")
                
                # Get epoch files for this condition
                epoch_files = get_epoch_files(base_path, condition_info['folder'])
                
                # Load epoch files
                epoch_data = {}
                for epoch_type, file_path in epoch_files.items():
                    print(f"Loading {epoch_type} epochs from {file_path}")
                    epochs = mne.read_epochs(file_path, preload=True)
                    epoch_data[epoch_type] = epochs

                if condition_name == "vep":
                    # Restore VEP plotting exactly as in original script
                    fig_vep, axes_vep = plt.subplots(2, 2, figsize=(15, 10))
                    fig_vep_inv, axes_vep_inv = plt.subplots(2, 2, figsize=(15, 10))
                    fig_vep.suptitle(f'{participant} - Visual Evoked Potentials', fontsize=12)
                    fig_vep_inv.suptitle(f'{participant} - Visual Evoked Potentials (Inverted)', fontsize=12)
                    
                    for i, group_name in enumerate(VEP_GROUPS):
                        try:
                            process_and_plot_vep(epoch_data, electrode_groups[group_name], axes_vep[i, 0], 
                                                 f'Stimulus-locked VEP ({group_name})', 'Stim', invert=False)
                            process_and_plot_vep(epoch_data, electrode_groups[group_name], axes_vep[i, 1], 
                                                 f'Outcome-locked VEP ({group_name})', 'Outcome', invert=False)
                            process_and_plot_vep(epoch_data, electrode_groups[group_name], axes_vep_inv[i, 0], 
                                                 f'Stimulus-locked VEP ({group_name})', 'Stim', invert=True)
                            process_and_plot_vep(epoch_data, electrode_groups[group_name], axes_vep_inv[i, 1], 
                                                 f'Outcome-locked VEP ({group_name})', 'Outcome', invert=True)
                        except Exception as e:
                            print(f"Warning: Could not process {group_name} VEP: {str(e)}")
                    
                    fig_vep.tight_layout()
                    fig_vep_inv.tight_layout()
                    
                    vep_fig_path = os.path.join(output_dir, f'{participant}_vep_analysis.png')
                    vep_fig_inv_path = os.path.join(output_dir, f'{participant}_vep_analysis_inverted.png')
                    fig_vep.savefig(vep_fig_path, dpi=300, bbox_inches='tight')
                    fig_vep_inv.savefig(vep_fig_inv_path, dpi=300, bbox_inches='tight')
                    
                    plt.close(fig_vep)
                    plt.close(fig_vep_inv)
                else:
                    # Create figures for normal and inverted polarity HEP analysis
                    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
                    fig1_inv, axes1_inv = plt.subplots(2, 2, figsize=(15, 10))
                    fig1.suptitle(f'{participant} - Combined HEP Analysis - {condition_name.upper()}', fontsize=12)
                    fig1_inv.suptitle(f'{participant} - Inverted HEP Analysis - {condition_name.upper()}', fontsize=12)
                    
                    for i, group_name in enumerate(HEP_GROUPS):
                        process_and_plot(epoch_data, electrode_groups[group_name], axes1[i, 0], 
                                         f'Outcome-locked HEP: {group_name}', condition_info, peak_type='Outcome')
                        process_and_plot(epoch_data, electrode_groups[group_name], axes1[i, 1], 
                                         f'Averaged R-peaks HEP: {group_name}', condition_info)
                        process_and_plot(epoch_data, electrode_groups[group_name], axes1_inv[i, 0], 
                                         f'Outcome-locked HEP: {group_name}', condition_info, peak_type='Outcome', invert=True)
                        process_and_plot(epoch_data, electrode_groups[group_name], axes1_inv[i, 1], 
                                         f'Averaged R-peaks HEP: {group_name}', condition_info, invert=True)
                    
                    fig1.tight_layout()
                    fig1_inv.tight_layout()
                    
                    fig1_path = os.path.join(output_dir, f'{participant}_{condition_name}_combined_hep_analysis.png')
                    fig1_inv_path = os.path.join(output_dir, f'{participant}_{condition_name}_combined_hep_analysis_inverted.png')
                    
                    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
                    fig1_inv.savefig(fig1_inv_path, dpi=300, bbox_inches='tight')
                    
                    plt.close(fig1)
                    plt.close(fig1_inv)
                
            print(f"\nCompleted analysis for {participant}")
            
        except Exception as e:
            print(f"Error processing {participant}: {str(e)}")
            continue

if __name__ == "__main__":
    main()