import numpy as np
import pandas as pd
from pathlib import Path
import mne
import pyxdf
import json
from scipy import stats
from mne.stats import permutation_cluster_test
import matplotlib.pyplot as plt
 
# Channel definitions
ALL_CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz',
    'AF1', 'AF2', 'FC1', 'FC2', 'CP1', 'CP2', 'PO1', 'PO2', 'FC5', 'FC6',
    'CP5', 'CP6', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF5', 'AF6',
    'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6',
    'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',
    'Fpz', 'FCz', 'CPz', 'ECG'
]

ELECTRODE_GROUPS = {
    'Frontocentral': ['F1', 'Fz', 'F2', 'FC1', 'FCz', 'FC2'],
    'Centroparietal': ['C1', 'Cz', 'C2', 'CP1', 'CPz', 'CP2'],
    'Both': ['F1', 'Fz', 'F2', 'FC1', 'FCz', 'FC2', 
             'C1', 'Cz', 'C2', 'CP1', 'CPz', 'CP2']
}

TIME_WINDOWS = {
    'signed_pe': (0.198, 0.252),
    'correct_incorrect': (0.250, 0.300),
    'absolute_pe': [(0.252, 0.292), (0.418, 0.464)]
}

def load_data():
    """Load all required data files."""
    # Setup paths
    base_path = Path(r"H:\Post\6th_Phase (ECG-EEG Baced BCI)")
    xdf_file = base_path / "2025.01.14/Pilot/Harry/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
    timing_file = base_path / "2025.01.17/Pilot Data Analysis/timing_analysis/trial_timing_analysis_20250117_150100.csv"
    model_file = base_path / "2025.01.17/Pilot Data Analysis/rl_model_results/Harry_rl_model_results_130445.json"
    
    print("Loading data files...")
    print(f"XDF file: {xdf_file}")
    print(f"Timing file: {timing_file}")
    print(f"Model file: {model_file}")
    
    # Verify files exist
    if not xdf_file.exists():
        raise FileNotFoundError(f"XDF file not found: {xdf_file}")
    if not timing_file.exists():
        raise FileNotFoundError(f"Timing file not found: {timing_file}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    # Load XDF
    streams, header = pyxdf.load_xdf(str(xdf_file))
    eeg_stream = streams[2]
    srate = float(eeg_stream['info']['nominal_srate'][0])
    eeg_data = eeg_stream['time_series']
    
    # Load timing data
    timing_df = pd.read_csv(timing_file)
    
    # Load model results
    with open(model_file, 'r') as f:
        model_data = json.load(f)
    
    print(f"Loaded EEG data: {eeg_data.shape[1]} channels at {srate} Hz")
    print(f"Loaded {len(timing_df)} trials with timing information")
    
    return eeg_data, srate, timing_df, model_data

def preprocess_eeg(eeg_data, srate):
    """Preprocess EEG data with proper handling of all 64 channels."""
    ch_types = ['eeg'] * 63 + ['ecg']
    
    info = mne.create_info(
        ch_names=ALL_CHANNEL_NAMES,
        sfreq=srate,
        ch_types=ch_types
    )
    
    raw = mne.io.RawArray(eeg_data.T, info)
    
    # Print annotations before preprocessing
    print("\nChecking for annotations:")
    print(raw.annotations)
    if len(raw.annotations) > 0:
        print("\nAnnotation details:")
        for ann in raw.annotations:
            print(f"Description: {ann['description']}, Start: {ann['onset']}, Duration: {ann['duration']}")
    else:
        print("No annotations found in raw data")
    
    # Apply minimal preprocessing
    raw.filter(l_freq=0.1, h_freq=40.0)  # More lenient high-pass
    raw.notch_filter(freqs=50, picks=['eeg'])  # Only apply notch to EEG channels
    
    print("\nPreprocessing complete:")
    print(f"- Data shape: {raw.get_data().shape}")
    print(f"- Number of channels: {len(raw.ch_names)}")
    print(f"- Sampling rate: {raw.info['sfreq']} Hz")
    
    return raw

def extract_manual_epochs(raw_data, srate, peak_times, channels, tmin=-0.2, tmax=0.6):
    """
    Manually extract epochs around R-peaks without using MNE Epochs.
    
    Parameters:
    -----------
    raw_data : np.array
        Raw EEG data array (channels x time)
    srate : float
        Sampling rate
    peak_times : array-like
        R-peak times in seconds
    channels : list
        List of channel indices to extract
    tmin, tmax : float
        Time window around peak in seconds
    """
    # Convert time window to samples
    samples_before = int(-tmin * srate)
    samples_after = int(tmax * srate)
    total_samples = samples_before + samples_after
    
    # Initialize epochs array
    epochs = []
    
    # Extract data around each peak
    for peak_time in peak_times:
        peak_sample = int(peak_time * srate)
        
        # Check if we have enough samples before and after
        if peak_sample - samples_before < 0 or peak_sample + samples_after > raw_data.shape[1]:
            continue
            
        # Extract epoch
        epoch = raw_data[channels, 
                        peak_sample - samples_before : peak_sample + samples_after]
        epochs.append(epoch)
    
    return np.array(epochs)

def get_channel_indices(channel_names, all_channels):
    """Get indices of channels in the full channel list."""
    return [all_channels.index(ch) for ch in channel_names]

def extract_condition_epochs(raw, timing_df, model_data, condition, rpeak_num, electrode_group):
    """Extract epochs for a specific condition and R-peak."""
    
    print(f"\nDebug info for {condition}, R-peak {rpeak_num}:")
    
    # Get trial indices based on condition
    if condition in ['positive_pe', 'negative_pe', 'high_abs_pe', 'low_abs_pe']:
        trial_indices = np.where(model_data['trial_classifications'][condition])[0]
    elif condition == 'correct':
        trial_indices = timing_df[timing_df['feedback'] == 'win'].index.values
    elif condition == 'incorrect':
        trial_indices = timing_df[timing_df['feedback'] == 'loss'].index.values
    else:
        raise ValueError(f"Unknown condition: {condition}")
    
    print(f"Number of trials in condition: {len(trial_indices)}")
    
    # Get corresponding R-peaks
    r_peaks = []
    peak_column = f'r{rpeak_num}_time'
    
    for idx in trial_indices:
        if idx < len(timing_df):
            peak_time = timing_df.iloc[idx][peak_column]
            if pd.notna(peak_time):
                r_peaks.append(float(peak_time))
    
    if not r_peaks:
        print(f"Warning: No R-peaks found for condition {condition}, peak {rpeak_num}")
        return None
    
    try:
        # Get channel indices
        channel_indices = get_channel_indices(ELECTRODE_GROUPS[electrode_group], ALL_CHANNEL_NAMES)
        
        # Extract epochs manually
        epochs = extract_manual_epochs(
            raw_data=raw.get_data(),
            srate=raw.info['sfreq'],
            peak_times=r_peaks,
            channels=channel_indices
        )
        
        print(f"Extracted {len(epochs)} epochs of shape {epochs.shape}")
        
        return epochs
        
    except Exception as e:
        print(f"Error during epoch creation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
def analyze_condition_pair(raw, timing_df, model_data, cond1, cond2, rpeak_num, 
                         electrode_group, time_window, output_dir):
    """Analyze a pair of conditions for a specific R-peak and electrode group."""
    print(f"\nAnalyzing {cond1} vs {cond2}")
    print(f"R-peak {rpeak_num}, Electrode group: {electrode_group}")
    
    try:
        # Extract epochs
        epochs1 = extract_condition_epochs(raw, timing_df, model_data, cond1, rpeak_num, electrode_group)
        epochs2 = extract_condition_epochs(raw, timing_df, model_data, cond2, rpeak_num, electrode_group)
        
        if epochs1 is None or epochs2 is None:
            print("Skipping analysis due to missing epochs")
            return
        
        if len(epochs1) == 0 or len(epochs2) == 0:
            print("Skipping analysis due to empty epochs")
            return
        
        # Print shapes for debugging
        print(f"Condition {cond1} epochs shape: {epochs1.shape}")
        print(f"Condition {cond2} epochs shape: {epochs2.shape}")
        
        # Analyze differences
        t_obs, clusters, p_vals = analyze_condition_difference(epochs1, epochs2, time_window)
        
        if t_obs is None:
            print("Statistical analysis failed")
            return
        
        # Save statistics
        stats = {
            'condition_pair': f"{cond1}_vs_{cond2}",
            'rpeak': rpeak_num,
            'electrode_group': electrode_group,
            'time_window': list(time_window),
            'shapes': {
                cond1: list(epochs1.shape),
                cond2: list(epochs2.shape)
            }
        }
        
        if p_vals is not None:
            stats['p_values'] = p_vals.tolist()
        if t_obs is not None:
            stats['t_values'] = t_obs.tolist()
            
        stats_file = output_dir / f"stats_{cond1}_vs_{cond2}_rpeak{rpeak_num}_{electrode_group}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
            
        # Create plots
        try:
            plot_comparison(epochs1, epochs2, raw.info['sfreq'], 
                          f"{cond1}_vs_{cond2}_rpeak{rpeak_num}_{electrode_group}",
                          output_dir)
        except Exception as e:
            print(f"Error in plotting: {str(e)}")
            
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_comparison(epochs1, epochs2, sfreq, name, output_dir):
    """Plot comparison between conditions."""
    times = np.linspace(-0.2, 0.6, epochs1.shape[2])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(name)
    
    for ch in range(min(6, epochs1.shape[1])):
        ax = axes[ch//3, ch%3]
        
        # Plot mean and SEM
        mean1 = epochs1[:, ch, :].mean(axis=0)
        sem1 = epochs1[:, ch, :].std(axis=0) / np.sqrt(epochs1.shape[0])
        mean2 = epochs2[:, ch, :].mean(axis=0)
        sem2 = epochs2[:, ch, :].std(axis=0) / np.sqrt(epochs2.shape[0])
        
        ax.fill_between(times, mean1-sem1, mean1+sem1, alpha=0.2, color='blue')
        ax.fill_between(times, mean2-sem2, mean2+sem2, alpha=0.2, color='red')
        ax.plot(times, mean1, 'blue', label='Condition 1')
        ax.plot(times, mean2, 'red', label='Condition 2')
        
        ax.axvline(x=0, color='k', linestyle='--')
        ax.grid(True)
        if ch == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{name}.png")
    plt.close()

def main():
    # Setup output directory
    output_dir = Path.cwd() / "hep_results"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    eeg_data, srate, timing_df, model_data = load_data()
    
    # Preprocess EEG
    raw = preprocess_eeg(eeg_data, srate)
    
    # Define conditions to analyze
    condition_pairs = [
        ('positive_pe', 'negative_pe', TIME_WINDOWS['signed_pe']),
        ('high_abs_pe', 'low_abs_pe', TIME_WINDOWS['absolute_pe'][0]),
        ('correct', 'incorrect', TIME_WINDOWS['correct_incorrect'])
    ]
    
    # Analyze each condition pair for each R-peak and electrode group
    for rpeak in range(1, 4):  # First three R-peaks
        for group in ['Frontocentral', 'Centroparietal', 'Both']:
            for cond1, cond2, time_window in condition_pairs:
                analyze_condition_pair(raw, timing_df, model_data, 
                                    cond1, cond2, rpeak, group, 
                                    time_window, output_dir)

if __name__ == "__main__":
    main()
