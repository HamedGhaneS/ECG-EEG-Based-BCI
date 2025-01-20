import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyxdf
from eeg_montage import CHANNEL_NAMES, CHANNEL_GROUPS, ANALYSIS_GROUPS


def load_timing_data(timing_file):
    """Load and parse the timing log file."""
    trials = []
    current_trial = {}

    with open(timing_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if 'Block' in line and 'Trial' in line:
            if current_trial:
                trials.append(current_trial)
            current_trial = {}

        if 'Absolute PE:' in line:
            try:
                pe_value = float(line.split('(')[1].split(')')[0])
                current_trial['abs_pe'] = pe_value
            except:
                continue

        if 'R1:' in line and 'ms' not in line:
            try:
                time = float(line.split(':')[1].split('s')[0])
                current_trial['r1_time'] = time
            except:
                continue

    if current_trial:
        trials.append(current_trial)

    return pd.DataFrame(trials)


def create_condition_events(timing_data, sfreq):
    """Create events for high and low absolute PE conditions."""
    # Remove any trials with missing data
    timing_data = timing_data.dropna(subset=['abs_pe', 'r1_time'])

    # Split trials into high and low PE based on median
    median_pe = timing_data['abs_pe'].median()
    high_pe_trials = timing_data[timing_data['abs_pe'] > median_pe]
    low_pe_trials = timing_data[timing_data['abs_pe'] <= median_pe]

    # Create events for both conditions
    events = []

    # High PE events (event_id = 1)
    for time in high_pe_trials['r1_time']:
        sample = int(time * sfreq)
        events.append([sample, 0, 1])

    # Low PE events (event_id = 2)
    for time in low_pe_trials['r1_time']:
        sample = int(time * sfreq)
        events.append([sample, 0, 2])

    events = np.array(sorted(events, key=lambda x: x[0]))

    # Debugging: print all events and check for duplicates
    print("All events created:")
    print(events)

    unique_samples = set()
    duplicates = []

    for event in events:
        if event[0] in unique_samples:
            duplicates.append(event)
        else:
            unique_samples.add(event[0])

    if duplicates:
        print("\nDuplicate events detected:")
        for dup in duplicates:
            print(f"Sample: {dup[0]}, Event ID: {dup[2]}")

    return events


def preprocess_raw(raw):
    """Apply preprocessing to the raw EEG data."""
    # Set ECG channel type
    raw.set_channel_types({'ECG': 'ecg'})

    # Apply bandpass filter
    raw.filter(0.1, 40.0)

    # Apply notch filter for line noise
    raw.notch_filter(np.arange(50, 251, 50))

    # Set EEG reference - using a simpler method
    raw.set_eeg_reference('average')

    return raw


def clean_with_ica(raw):
    """Apply ICA to remove cardiac artifacts."""
    from mne.preprocessing import ICA

    ica = ICA(n_components=20, random_state=42, method='fastica')

    # Find which channels to use for ICA fitting
    picks = mne.pick_types(raw.info, eeg=True, ecg=False)

    # Fit ICA on EEG channels
    ica.fit(raw, picks=picks)

    # Find ECG artifacts using the ECG channel as reference
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='ECG')

    if ecg_indices:
        ica.exclude = ecg_indices
        print(f"Found {len(ecg_indices)} ECG components")

    # Apply ICA correction
    raw_clean = raw.copy()
    ica.apply(raw_clean)

    return raw_clean


def analyze_hep(raw, events, timing_data):
    """Main HEP analysis function."""
    # Define event ids for conditions
    event_id = {'high_pe': 1, 'low_pe': 2}

    print(f"Creating epochs from {len(events)} events...")

    # Create epochs
    try:
        epochs = mne.Epochs(raw, events, event_id=event_id,
                            tmin=-0.2, tmax=0.6,  # Time window from paper
                            baseline=(-0.15, -0.05),  # Baseline period from paper
                            picks='eeg',
                            reject=None,  # Start without rejection
                            preload=True)

        print(f"Created {len(epochs)} total epochs")
    except Exception as e:
        print("Error while creating epochs:", e)
        print("Events causing the error:")
        print(events)
        raise e

    # Use predefined centroparietal channels from montage
    centro_parietal_chs = ANALYSIS_GROUPS['Centroparietal']

    # Calculate ERPs for each condition
    high_pe_erp = epochs['high_pe'].average()
    low_pe_erp = epochs['low_pe'].average()

    # Perform cluster-based permutation test
    adjacency, ch_names = mne.channels.find_ch_adjacency(epochs.info, ch_type='eeg')

    # Time window for statistical test
    times_mask = np.logical_and(epochs.times >= 0.1, epochs.times <= 0.5)

    stat_data_high = epochs['high_pe'].get_data()[:, :, times_mask]
    stat_data_low = epochs['low_pe'].get_data()[:, :, times_mask]

    t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(
        [stat_data_high, stat_data_low],
        n_permutations=1000,
        adjacency=adjacency,
        tail=0)

    return high_pe_erp, low_pe_erp, t_obs, clusters, cluster_pv, epochs.times[times_mask]


def plot_results(high_pe_erp, low_pe_erp, times, significant_times=None):
    """Plot HEP results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot ERPs
    times_ms = times * 1000  # Convert to milliseconds
    high_data = high_pe_erp.data.mean(axis=0)
    low_data = low_pe_erp.data.mean(axis=0)

    ax.plot(times_ms, high_data, 'r', label='High AbsPE')
    ax.plot(times_ms, low_data, 'b', label='Low AbsPE')

    # Plot significant time periods
    if significant_times is not None:
        ax.fill_between(times_ms[significant_times],
                        high_data[significant_times],
                        low_data[significant_times],
                        color='gray', alpha=0.3)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()
    ax.set_title('HEP by Absolute PE Condition')

    plt.tight_layout()
    return fig


def main():
    """
    Main pipeline for HEP analysis with detailed debugging logs.
    """
    # File paths
    raw_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry\sub-P001\ses-S001\eeg\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
    timing_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.20\Pilot Data Analysis\timing_analysis\trial_timing_log_with_removed.txt"

    # Load XDF data
    print("Loading XDF file...")
    streams, header = pyxdf.load_xdf(raw_file)

    eeg_stream = streams[2]  # Assume the third stream is EEG
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    eeg_data = np.array(eeg_stream['time_series']).T

    print("Creating MNE Raw object...")
    raw = mne.io.RawArray(eeg_data, mne.create_info(CHANNEL_NAMES, sfreq, ch_types='eeg'))

    print("Loading timing data...")
    timing_data = load_timing_data(timing_file)

    print("Preprocessing data...")
    raw = preprocess_raw(raw)

    print("Cleaning data with ICA...")
    raw_clean = clean_with_ica(raw)

    print("Creating events...")
    events = create_condition_events(timing_data, sfreq)

    print("Analyzing HEP...")
    high_pe_erp, low_pe_erp, t_obs, clusters, cluster_pv, times = analyze_hep(raw_clean, events, timing_data)

    print("Plotting results...")
    plot_results(high_pe_erp, low_pe_erp, times)


if __name__ == "__main__":
    main()
