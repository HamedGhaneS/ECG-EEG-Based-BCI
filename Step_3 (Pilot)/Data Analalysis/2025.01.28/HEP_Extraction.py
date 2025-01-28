"""
# EEG-HEP Analysis Script
# Author: Hamed GHane
# Date: January 28, 2025

# Workflow Overview:
1. **Dependencies:**
   - Import libraries: `mne`, `numpy`, `matplotlib`, `scipy`, `os`.

2. **Electrode Groups and Conditions:**
   - Define electrode groups (e.g., `Frontocentral`, `Centroparietal`).
   - Map experimental conditions to event types and file paths.

3. **Preprocessing Functions:**
   - `find_continuous_periods`: Identify significant time periods (e.g., >15 ms).
   - `process_and_plot`: Perform baseline correction and HEP analysis:
       - **Baseline correction:** (-0.2 to -0.05 seconds).
       - **Averaging:** Combine trials or R-peaks (R1, R2, R3).
   - `plot_results`: Plot data with statistical overlays:
       - Calculate mean, SEM, and significant time periods (p < 0.05).
   - `get_participant_paths`: Retrieve base paths for participant-specific data.
   - `get_epoch_files`: Construct file paths for epochs.

4. **Interactive Workflow:**
   - Prompt user for participant name (`Elmira` or `Harry`).
   - Verify and load participant data.

5. **Data Preprocessing:**
   - **Epoch loading:** Load R-locked and Outcome-locked data.
   - **Channel selection:** Pick channels for analysis.
   - **Baseline correction:** Normalize data.

6. **Statistical Analysis:**
   - Perform t-tests between conditions.
   - Identify significant time windows (p < 0.05).
   - Label continuous significant periods (≥15 ms).

7. **Visualization:**
   - Generate and save:
       - Combined HEP analysis (Outcome-locked and Averaged R-peaks).
       - Individual R-peak analysis (R1, R2, R3) per electrode group.
   - Add mean, SEM, and significance markers to plots.

8. **Output Management:**
   - Create output directories for figures.
   - Save high-resolution plots for each condition.

9. **Program Flow:**
   - Loop through participants.
   - Process data for each condition (e.g., `abs_pe`, `pe_sign`).
   - Handle errors during data loading, processing, and saving.

# Note:
- This script processes EEG-HEP data for participant-specific real-time BCI experiments.
"""



import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Define groups of electrodes using channel names
electrode_groups = {
    "Frontocentral": ['F1', 'F2', 'Fz', 'FC1', 'FC2', 'FCz'],
    "Centroparietal": ['C1', 'C2', 'Cz', 'CP1', 'CP2', 'CPz']
}

# Define conditions and their event mappings (excluding feedback)
analysis_conditions = {
    "abs_pe": {
        "folder": "abs_pe_epochs",
        "events": {"high_pe": 1, "low_pe": 2}
    },
    "pe_sign": {
        "folder": "pe_sign_epochs",
        "events": {"positive_pe": 1, "negative_pe": 2}
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
        epochs = epochs_dict[peak_type].copy()  # Create a copy to avoid modifying original
    else:
        if peak_type == 'Outcome':
            epochs = epochs_dict['Outcome'].copy()
        else:
            # Average across R-peaks
            cond1_list = []
            cond2_list = []
            for peak in ['R1', 'R2', 'R3']:
                peak_epochs = epochs_dict[peak].copy()
                peak_epochs.pick_channels(channels)
                peak_epochs.apply_baseline(baseline=(-0.2, -0.05))
                
                cond1_list.append(peak_epochs[event_names[0]].get_data().mean(axis=1))
                cond2_list.append(peak_epochs[event_names[1]].get_data().mean(axis=1))
            
            cond1_data = np.mean(cond1_list, axis=0)
            cond2_data = np.mean(cond2_list, axis=0)
            times = peak_epochs.times
            print(f"\nAveraged R-peaks analysis for {title}")
            return plot_results(cond1_data, cond2_data, times, ax, title, event_names)

    # For single peak or outcome analysis
    epochs.pick_channels(channels)
    epochs.apply_baseline(baseline=(-0.2, -0.05))
    
    cond1_data = epochs[event_names[0]].get_data().mean(axis=1)
    cond2_data = epochs[event_names[1]].get_data().mean(axis=1)
    times = epochs.times
    
    print(f"\nAnalysis for {title}")
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
            
            # Process each condition separately
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
                
                # Create figures for this condition
                fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
                fig1.suptitle(f'{participant} - Combined HEP Analysis - {condition_name.upper()}', fontsize=12)
                
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
                fig2.suptitle(f'{participant} - Individual R-peak HEP Analysis - {condition_name.upper()}', fontsize=12)
                
                # Plot individual R-peak analyses
                for i, (group_name, channels) in enumerate(electrode_groups.items()):
                    for j, peak in enumerate(['R1', 'R2', 'R3']):
                        process_and_plot(epoch_data, channels, axes2[i, j],
                                       f'{peak}-locked HEP: {group_name}',
                                       condition_info, is_single_peak=True, peak_type=peak)
                
                fig2.tight_layout()
                
                # Save figures with debugging information
                output_dir = os.path.join(base_path, 'HEP_Analysis_plot')
                print(f"\nTrying to create directory at: {output_dir}")
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"Directory created/verified at: {output_dir}")
                except Exception as e:
                    print(f"Error creating directory: {str(e)}")
                    if os.path.exists(output_dir):
                        print("Directory already exists")
                    else:
                        print("Directory does not exist and could not be created")
                        continue
                
                # Save the figures
                try:
                    fig1_path = os.path.join(output_dir, f'{participant}_{condition_name}_combined_hep_analysis.png')
                    fig2_path = os.path.join(output_dir, f'{participant}_{condition_name}_individual_peaks_analysis.png')
                    
                    print(f"Saving combined analysis to: {fig1_path}")
                    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
                    
                    print(f"Saving individual peaks analysis to: {fig2_path}")
                    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
                    
                except Exception as e:
                    print(f"Error saving figures: {str(e)}")
                
                plt.close(fig1)
                plt.close(fig2)
                
            print(f"\nCompleted analysis for {participant}")
            
        except Exception as e:
            print(f"Error processing {participant}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
