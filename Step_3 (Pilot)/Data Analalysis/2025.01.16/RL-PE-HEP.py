"""
HEP Analysis Script for Cardiac Learning Study
--------------------------------------------
This script analyzes Heartbeat Evoked Potentials (HEP) following Figure 2 of the Nature 
Communications paper, handling 64-channel EEG data from XDF files. It examines HEP 
differences between conditions for the first three R-peaks after outcome presentation.

Analysis conditions:
- Positive vs Negative signed PE (Frontocentral electrodes)
- Correct vs Incorrect outcomes (Centroparietal electrodes)
- High vs Low absolute PE (Centroparietal electrodes)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import mne
import pyxdf
import json
from scipy import stats
from mne.stats import permutation_cluster_test
import matplotlib.pyplot as plt

# Define the complete channel montage
ALL_CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
    'AF1', 'AF2', 'FC1', 'FC2', 'CP1', 'CP2', 'PO1', 'PO2', 'FC5', 'FC6',
    'CP5', 'CP6', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF5', 'AF6',
    'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6',
    'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',
    'Fpz', 'FCz', 'CPz', 'ECG'
]

# Define the specific electrode groups for analysis (from Figure 2)
PAPER_ANALYSIS_GROUPS = {
    'Frontocentral': ['F1', 'Fz', 'F2', 'FC1', 'FCz', 'FC2'],
    'Centroparietal': ['C1', 'Cz', 'C2', 'CP1', 'CPz', 'CP2']
}

# Time windows from the paper for different analyses
TIME_WINDOWS = {
    'signed_pe': (0.198, 0.252),
    'correct_incorrect': (0.250, 0.300),
    'absolute_pe': [(0.252, 0.292), (0.418, 0.464)]
}

def load_xdf_data(file_path):
    """Load EEG/ECG data from XDF file."""
    print(f"Loading XDF file: {file_path}")
    streams, header = pyxdf.load_xdf(file_path)
    
    # Get EEG stream (third stream in the XDF file)
    eeg_stream = streams[2]
    srate = float(eeg_stream['info']['nominal_srate'][0])
    eeg_data = eeg_stream['time_series']
    
    print(f"Loaded EEG data: {eeg_data.shape[1]} channels at {srate} Hz")
    return eeg_data, srate, streams

def preprocess_eeg(eeg_data, srate):
    """
    Preprocess EEG data with proper handling of all 64 channels.
    Applies filtering and creates an MNE Raw object.
    """
    # Create channel types array
    ch_types = ['eeg'] * 63 + ['ecg']
    
    # Create MNE info object
    info = mne.create_info(
        ch_names=ALL_CHANNEL_NAMES,
        sfreq=srate,
        ch_types=ch_types
    )
    
    # Create Raw object with transposed data to match MNE's format
    raw = mne.io.RawArray(eeg_data.T, info)
    
    # Apply standard preprocessing
    raw.filter(l_freq=0.5, h_freq=40.0)
    raw.notch_filter(freqs=50)
    
    print("\nPreprocessing complete:")
    print(f"- Data shape: {raw.get_data().shape}")
    print(f"- Number of channels: {len(raw.ch_names)}")
    print(f"- Sampling rate: {raw.info['sfreq']} Hz")
    
    return raw

def extract_r_peaks(raw, srate):
    """
    Detect R-peaks in ECG data using MNE's ecg detection.
    Returns R-peak sample indices.
    """
    # Get ECG data
    ecg_data = raw.get_data(picks=['ECG'])
    
    # Find ECG events
    events, _, _ = mne.preprocessing.find_ecg_events(
        raw,
        ch_name='ECG',
        event_id=999,
        sampling_rate=srate
    )
    
    return events[:, 0]  # Return sample indices of R-peaks

def find_post_outcome_peaks(r_peaks, outcome_times, srate, n_peaks=3):
    """Find the first three R-peaks after each outcome."""
    post_outcome_peaks = []
    
    for outcome_time in outcome_times:
        outcome_sample = int(outcome_time * srate)
        peaks_after = r_peaks[r_peaks > outcome_sample]
        
        if len(peaks_after) >= n_peaks:
            post_outcome_peaks.append(peaks_after[:n_peaks])
    
    return np.array(post_outcome_peaks)

def extract_hep_epochs(raw, r_peaks, group_name, tmin=-0.2, tmax=0.6):
    """
    Extract HEP epochs around R-peaks for specified electrode group.
    Uses MNE's epoch extraction functionality.
    """
    # Create events array for MNE
    events = np.column_stack([r_peaks, np.zeros_like(r_peaks), np.ones_like(r_peaks)])
    
    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=1,
        tmin=tmin,
        tmax=tmax,
        picks=PAPER_ANALYSIS_GROUPS[group_name],
        baseline=(tmin, -0.05),
        preload=True
    )
    
    return epochs.get_data()

def analyze_condition_difference(epochs1, epochs2, time_window):
    """Perform cluster-based permutation test between conditions."""
    # Convert time window to samples
    srate = 5000  # Your sampling rate
    start_sample = int((time_window[0] + 0.2) * srate)
    end_sample = int((time_window[1] + 0.2) * srate)
    
    # Extract relevant time window
    data1 = epochs1[:, :, start_sample:end_sample]
    data2 = epochs2[:, :, start_sample:end_sample]
    
    # Perform cluster test
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_test(
        [data1, data2],
        n_permutations=1000,
        threshold=2.0,
        tail=0
    )
    
    return t_obs, clusters, cluster_pv

def plot_hep_results(epochs1, epochs2, times, group_name, condition_name, outpath):
    """Create plots similar to Figure 2 of the paper."""
    mean1 = epochs1.mean(axis=0)
    mean2 = epochs2.mean(axis=0)
    sem1 = epochs1.std(axis=0) / np.sqrt(epochs1.shape[0])
    sem2 = epochs2.std(axis=0) / np.sqrt(epochs2.shape[0])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    channel_names = PAPER_ANALYSIS_GROUPS[group_name]
    for idx, (ch_name, ax) in enumerate(zip(channel_names, axes)):
        ax.plot(times, mean1[idx], 'r', label='Condition 1')
        ax.fill_between(times, mean1[idx]-sem1[idx], mean1[idx]+sem1[idx], 
                       color='r', alpha=0.2)
        ax.plot(times, mean2[idx], 'b', label='Condition 2')
        ax.fill_between(times, mean2[idx]-sem2[idx], mean2[idx]+sem2[idx], 
                       color='b', alpha=0.2)
        ax.axvline(x=0, color='k', linestyle='--')
        ax.set_title(ch_name)
        if idx == 0:
            ax.legend()
    
    plt.suptitle(f'{condition_name} - {group_name} electrodes')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    # Set up paths
    base_path = Path(r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry")
    xdf_file = base_path / "sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
    model_file = base_path / "rl_model_results/Harry_rl_model_results_130445.json"
    output_dir = Path(__file__).parent / "hep_analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    # Load and preprocess data
    eeg_data, srate, streams = load_xdf_data(xdf_file)
    raw = preprocess_eeg(eeg_data, srate)
    
    # Load model results
    with open(model_file, 'r') as f:
        model_results = json.load(f)
    trial_info = model_results['trial_classifications']
    
    # Get R-peaks
    r_peaks = extract_r_peaks(raw, srate)
    
    # Define analysis conditions
    conditions = [
        ('positive_pe', 'negative_pe', 'Frontocentral', TIME_WINDOWS['signed_pe']),
        ('correct', 'incorrect', 'Centroparietal', TIME_WINDOWS['correct_incorrect']),
        ('high_abs_pe', 'low_abs_pe', 'Centroparietal', TIME_WINDOWS['absolute_pe'][0])
    ]
    
    # Analyze each R-peak
    for peak_num in range(3):
        print(f"\nAnalyzing R-peak {peak_num + 1}")
        
        # Get outcome times from your data (you'll need to implement this based on your markers)
        outcome_times = []  # Fill this with actual outcome times
        
        # Get post-outcome R-peaks
        post_outcome_peaks = find_post_outcome_peaks(r_peaks, outcome_times, srate)
        
        for cond1, cond2, group, time_window in conditions:
            print(f"\nAnalyzing {cond1} vs {cond2} in {group} electrodes")
            
            # Extract epochs for each condition
            epochs_cond1 = extract_hep_epochs(
                raw, 
                post_outcome_peaks[np.array(trial_info[cond1]), peak_num],
                group
            )
            epochs_cond2 = extract_hep_epochs(
                raw,
                post_outcome_peaks[np.array(trial_info[cond2]), peak_num],
                group
            )
            
            # Analyze differences
            t_obs, clusters, p_vals = analyze_condition_difference(
                epochs_cond1,
                epochs_cond2,
                time_window
            )
            
            # Save results
            results = {
                'condition': f"{cond1} vs {cond2}",
                'peak_number': peak_num + 1,
                'electrode_group': group,
                'time_window': time_window,
                'cluster_p_values': p_vals.tolist(),
                't_statistics': t_obs.tolist()
            }
            
            results_file = output_dir / f"stats_{cond1}_vs_{cond2}_peak{peak_num+1}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            
            # Create and save plots
            plot_file = output_dir / f"hep_{cond1}_vs_{cond2}_peak{peak_num+1}.png"
            times = np.linspace(-0.2, 0.6, epochs_cond1.shape[2])
            plot_hep_results(
                epochs_cond1,
                epochs_cond2,
                times,
                group,
                f"{cond1} vs {cond2}",
                plot_file
            )

if __name__ == "__main__":
    main()