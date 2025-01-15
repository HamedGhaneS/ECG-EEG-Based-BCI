import pyxdf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import stats
import pandas as pd
from pathlib import Path
from itertools import combinations

def load_experiment_data(base_path):
    """Load and validate all experimental log files."""
    print("Loading experiment data...")
    
    # Define file paths
    exp_file = base_path / "Harry-ses001-run1-20250114-130445.csv"
    timing_file = base_path / "Harry-ses001-run1-20250114-130445_timing_pairs.csv"
    block_file = base_path / "Harry-ses001-run1-20250114-130445_block_order.txt"
    
    print(f"Loading main data from: {exp_file}")
    print(f"Loading timing data from: {timing_file}")
    print(f"Loading block data from: {block_file}")
    
    try:
        # Load main experiment data and examine its structure
        exp_data = pd.read_csv(exp_file)
        print("\nExperiment data summary:")
        print(f"Shape: {exp_data.shape}")
        print("Columns:", exp_data.columns.tolist())
        
        # Examine R-peak timing data
        rpeak_times = exp_data['valid_rpeak_time'].dropna()
        print("\nR-peak timing summary:")
        print(f"Total R-peaks: {len(rpeak_times)}")
        print(f"Time range: {rpeak_times.min():.3f}s to {rpeak_times.max():.3f}s")
        
        # Load timing pairs
        timing_pairs = pd.read_csv(timing_file)
        print(f"\nTiming pairs shape: {timing_pairs.shape}")
        
        # Parse block information from text file
        with open(block_file, 'r') as f:
            block_order = f.read()
            
        # Extract block information
        block_info = {}
        current_block = None
        for line in block_order.split('\n'):
            if 'Block' in line and ':' in line:
                try:
                    current_block = int(line.split(':')[0].split()[-1])
                    block_info[current_block] = {}
                except ValueError:
                    continue
            elif current_block is not None:
                if 'Cardiac Phase:' in line:
                    block_info[current_block]['phase'] = line.split(':')[1].strip()
                elif 'Timing Point:' in line:
                    block_info[current_block]['timing'] = line.split(':')[1].strip()
                elif 'R-R Percentage:' in line:
                    try:
                        percentage = float(line.split(':')[1].strip().replace('%', '')) / 100
                        block_info[current_block]['percentage'] = percentage
                    except ValueError:
                        continue
        
        print("\nBlock Information:")
        for block, info in block_info.items():
            print(f"Block {block}: {info}")
            
        return exp_data, timing_pairs, block_info
        
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        raise

def preprocess_eeg_data(eeg_data, fs):
    """Preprocess EEG data with bandpass filtering."""
    print("\nPreprocessing EEG data...")
    print(f"Input data shape: {eeg_data.shape}")
    
    nyq = fs / 2
    b_eeg, a_eeg = butter(1, [0.5/nyq, 40/nyq], btype='band')
    eeg_filtered = filtfilt(b_eeg, a_eeg, eeg_data, axis=0)
    
    print(f"Filtered data shape: {eeg_filtered.shape}")
    return eeg_filtered

def perform_condition_comparison(data1, data2, label1, label2, window_name=None):
    """Perform statistical comparison between two conditions."""
    if len(data1) == 0 or len(data2) == 0:
        print(f"Insufficient data for comparison between {label1} and {label2}")
        return
        
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    sem1 = stats.sem(data1)
    sem2 = stats.sem(data2)
    
    # Perform t-test
    t_stat, p_val = stats.ttest_ind(data1, data2)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(data1) - 1) * np.std(data1)**2 + 
                         (len(data2) - 1) * np.std(data2)**2) / 
                        (len(data1) + len(data2) - 2))
    cohens_d = (mean2 - mean1) / pooled_std
    
    window_str = f" ({window_name})" if window_name else ""
    print(f"\nComparison: {label1} vs {label2}{window_str}")
    print(f"{label1}: {mean1:.3f} ± {sem1:.3f} μV (n={len(data1)})")
    print(f"{label2}: {mean2:.3f} ± {sem2:.3f} μV (n={len(data2)})")
    print(f"Difference: {mean2 - mean1:.3f} μV")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_val:.4f}")
    print(f"Cohen's d: {cohens_d:.3f}")
    
    return {'t_stat': t_stat, 'p_val': p_val, 'cohens_d': cohens_d}

def extract_phase_epochs(eeg_data, exp_data, block_info, fs, tmin=-0.2, tmax=0.6):
    """Extract epochs based on cardiac phase and timing."""
    print("\nExtracting epochs by cardiac phase...")
    
    # Calculate epoch parameters
    n_samples = int((tmax - tmin) * fs)
    n_channels = eeg_data.shape[1] - 1  # Exclude ECG channel
    times = np.linspace(tmin, tmax, n_samples)
    
    print(f"\nEpoch extraction parameters:")
    print(f"EEG data shape: {eeg_data.shape}")
    print(f"Samples per epoch: {n_samples}")
    print(f"Number of channels: {n_channels}")
    print(f"Time window: {tmin}s to {tmax}s")
    
    # Calculate timing alignment
    first_rpeak = exp_data['valid_rpeak_time'].min()
    time_offset = first_rpeak
    print(f"\nTiming alignment:")
    print(f"First R-peak time: {first_rpeak:.3f}s")
    print(f"Using time offset: {time_offset:.3f}s")
    
    # Initialize conditions dictionary
    conditions = {
        'systole': {'early': [], 'mid': [], 'late': []},
        'diastole': {'early': [], 'mid': [], 'late': []}
    }
    
    # Process each block
    for block in range(1, 7):
        print(f"\nProcessing block {block}...")
        
        # Get block data
        block_mask = exp_data['block'] == (block - 1)  # Blocks are 0-indexed in data
        block_data = exp_data[block_mask]
        
        if block not in block_info:
            print(f"Warning: Block {block} not found in block info")
            continue
        
        phase = block_info[block]['phase']
        timing = block_info[block]['timing']
        
        # Get valid R-peaks for this block
        valid_peaks = block_data['valid_rpeak_time'].dropna().values
        print(f"\nFound {len(valid_peaks)} valid R-peaks")
        
        if len(valid_peaks) > 0:
            print(f"R-peak time range: {valid_peaks.min():.3f}s to {valid_peaks.max():.3f}s")
        
        # Process each R-peak
        successful_extractions = 0
        for rpeak_time in valid_peaks:
            # Adjust timing relative to EEG recording start
            adjusted_time = rpeak_time - time_offset
            
            # Convert to samples
            start_idx = int(adjusted_time * fs) + int(tmin * fs)
            end_idx = start_idx + n_samples
            
            if start_idx >= 0 and end_idx < len(eeg_data):
                try:
                    epoch = eeg_data[start_idx:end_idx, :n_channels]
                    if np.all(np.isfinite(epoch)):
                        conditions[phase][timing].append(epoch)
                        successful_extractions += 1
                except Exception as e:
                    print(f"Error extracting epoch: {str(e)}")
        
        print(f"Successfully extracted {successful_extractions} epochs")
    
    # Print final summary
    print("\nFinal epoch count summary:")
    for phase in ['systole', 'diastole']:
        for timing in ['early', 'mid', 'late']:
            print(f"{phase}-{timing}: {len(conditions[phase][timing])} epochs")
    
    return conditions, times

def analyze_cardiac_phases(conditions, times, channel_groups):
    """Analyze HEP differences between and within cardiac phases."""
    print("\nAnalyzing cardiac phase differences...")
    
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # Define colors and line styles
    colors = {
        'systole': {'early': 'r', 'mid': 'g', 'late': 'b'},
        'diastole': {'early': 'r--', 'mid': 'g--', 'late': 'b--'}
    }
    
    # Plot for each channel group
    for group_idx, (group_name, channels) in enumerate(channel_groups.items()):
        plt.subplot(2, 1, group_idx + 1)
        
        # Store mean values for statistical analysis
        phase_data = {'systole': {}, 'diastole': {}}
        
        for phase in ['systole', 'diastole']:
            for timing in ['early', 'mid', 'late']:
                epochs = conditions[phase][timing]
                
                if epochs:
                    print(f"\nProcessing {phase}-{timing} for {group_name}")
                    print(f"Number of epochs: {len(epochs)}")
                    
                    epochs = np.array(epochs)
                    # Average across channels within the group
                    channel_data = np.mean(epochs[:, :, channels], axis=2)
                    phase_data[phase][timing] = channel_data
                    
                    # Calculate mean and SEM for plotting
                    mean = np.mean(channel_data, axis=0)
                    sem = stats.sem(channel_data, axis=0)
                    
                    plt.plot(times * 1000, mean, colors[phase][timing], 
                            label=f'{phase}-{timing} (n={len(epochs)})')
                    plt.fill_between(times * 1000, mean-sem, mean+sem, alpha=0.2)
        
        plt.axvline(x=0, color='k', linestyle=':', alpha=0.5)
        plt.title(f'HEP by Cardiac Phase - {group_name}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (μV)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical analysis
    print("\nComprehensive Statistical Analysis")
    time_windows = {
        'Early': (0.1, 0.2),
        'Mid': (0.2, 0.3),
        'Late': (0.3, 0.4)
    }
    
    for window_name, (tmin, tmax) in time_windows.items():
        print(f"\n{window_name} Window ({int(tmin*1000)}-{int(tmax*1000)}ms):")
        window_mask = (times >= tmin) & (times <= tmax)
        
        for group_name, channels in channel_groups.items():
            print(f"\n{group_name} channels:")
            
            # Compare systole vs diastole (overall)
            print("\nOverall Phase Comparison:")
            systole_all = []
            diastole_all = []
            
            for timing in ['early', 'mid', 'late']:
                if timing in phase_data['systole'] and timing in phase_data['diastole']:
                    systole_data = np.mean(phase_data['systole'][timing][:, window_mask], axis=1)
                    diastole_data = np.mean(phase_data['diastole'][timing][:, window_mask], axis=1)
                    
                    systole_all.extend(systole_data)
                    diastole_all.extend(diastole_data)
            
            perform_condition_comparison(np.array(systole_all), np.array(diastole_all),
                                      'Systole', 'Diastole', window_name)
            
            # Within-phase comparisons
            print("\nWithin-Phase Comparisons:")
            for phase in ['systole', 'diastole']:
                print(f"\n{phase.capitalize()} Timing Comparisons:")
                timing_points = ['early', 'mid', 'late']
                
                for t1, t2 in combinations(timing_points, 2):
                    if t1 in phase_data[phase] and t2 in phase_data[phase]:
                        data1 = np.mean(phase_data[phase][t1][:, window_mask], axis=1)
                        data2 = np.mean(phase_data[phase][t2][:, window_mask], axis=1)
                        
                        perform_condition_comparison(data1, data2,
                                                  f"{phase}-{t1}",
                                                  f"{phase}-{t2}",
                                                  window_name)

def main():
    """Main function to run the complete analysis pipeline."""
    # Set up paths
    base_path = Path(r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry")
    xdf_path = base_path / "sub-P001" / "ses-S001" / "eeg" / "sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
    
    # Load experimental data
    exp_data, timing_pairs, block_info = load_experiment_data(base_path)
    
    # Load XDF file
    print("\nLoading XDF file...")
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg_stream = streams[2]  # Third stream containing EEG/ECG
    fs = float(eeg_stream['info']['nominal_srate'][0])
    print(f"Sampling rate: {fs} Hz")
    
    # Preprocess EEG data
    eeg_data = preprocess_eeg_data(eeg_stream['time_series'], fs)
    
    # Define channel groups
    channel_groups = {
        'Frontocentral': list(range(0, 6)),
        'Centroparietal': list(range(31, 37))
    }
    
    # Extract epochs by condition
    conditions, times = extract_phase_epochs(eeg_data, exp_data, block_info, fs)
    
    # Analyze differences
    analyze_cardiac_phases(conditions, times, channel_groups)

if __name__ == "__main__":
    main()