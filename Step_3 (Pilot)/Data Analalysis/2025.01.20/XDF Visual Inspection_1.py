import pyxdf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from eeg_montage import get_group_indices, CHANNEL_NAMES, CHANNEL_GROUPS

# Define electrode clusters
FRONTOCENTRAL = CHANNEL_GROUPS['Frontocentral']
CENTROPARIETAL = CHANNEL_GROUPS['Centroparietal']

# Function to load EEG data from XDF file
def load_eeg_data(file_path):
    """
    Load EEG data from an XDF file.

    Args:
        file_path (str): Path to the XDF file.

    Returns:
        np.ndarray: EEG data (channels x time points).
        float: Sampling frequency.
        list: Channel names.
    """
    streams, _ = pyxdf.load_xdf(file_path)
    eeg_stream = streams[2]  # Assume the third stream is EEG
    sfreq = float(eeg_stream['info']['nominal_srate'][0])  # Sampling frequency
    eeg_data = np.array(eeg_stream['time_series']).T  # Transpose to channels x time
    return eeg_data, sfreq, CHANNEL_NAMES

# Function to get data for specific electrode clusters
def get_cluster_data(eeg_data, cluster_names, channel_names):
    """
    Extract data for specific electrode clusters.

    Args:
        eeg_data (np.ndarray): EEG data (channels x time points).
        cluster_names (list): List of electrode names in the cluster.
        channel_names (list): List of all channel names.

    Returns:
        np.ndarray: EEG data for the specified cluster (channels x time points).
        list: Cluster channel names.
    """
    cluster_indices = [channel_names.index(name) for name in cluster_names]
    return eeg_data[cluster_indices, :], cluster_names

# Function to plot EEG channels one by one
def plot_eeg_channels(eeg_data, channel_names):
    """
    Plot EEG channels one by one.

    Args:
        eeg_data (np.ndarray): EEG data (channels x time points).
        channel_names (list): List of channel names.
    """
    for i, channel in enumerate(channel_names):
        plt.figure(figsize=(10, 5))
        plt.plot(eeg_data[i, :], color='blue', linewidth=0.7)
        plt.title(f"Channel {i + 1}: {channel}", fontsize=14)
        plt.xlabel("Time Points", fontsize=12)
        plt.ylabel("Amplitude (µV)", fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.show()  # Display the plot

# Function to compute channel statistics
def compute_channel_statistics(eeg_data, channel_names):
    """
    Compute basic statistics (mean, std, min, max) for each EEG channel.

    Args:
        eeg_data (np.ndarray): EEG data (channels x time points).
        channel_names (list): List of channel names.

    Returns:
        pd.DataFrame: DataFrame with statistics for each channel.
    """
    stats = {
        "Channel": channel_names,
        "Mean (µV)": eeg_data.mean(axis=1),
        "Std (µV)": eeg_data.std(axis=1),
        "Min (µV)": eeg_data.min(axis=1),
        "Max (µV)": eeg_data.max(axis=1),
    }
    stats_df = pd.DataFrame(stats)
    return stats_df

# Main function
def main():
    # File path to the XDF file
    xdf_path = r"H:\\Post\\6th_Phase (ECG-EEG Baced BCI)\\2025.01.14\\Pilot\\Harry\\sub-P001\\ses-S001\\eeg\\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"

    # Load EEG data
    print("Loading EEG data...")
    eeg_data, sfreq, channel_names = load_eeg_data(xdf_path)
    print(f"Loaded EEG data with {eeg_data.shape[0]} channels and {eeg_data.shape[1]} time points.")

    # Extract and plot Frontocentral cluster
    print("Processing Frontocentral cluster...")
    fc_data, fc_names = get_cluster_data(eeg_data, FRONTOCENTRAL, channel_names)
    plot_eeg_channels(fc_data, fc_names)

    # Extract and plot Centroparietal cluster
    print("Processing Centroparietal cluster...")
    cp_data, cp_names = get_cluster_data(eeg_data, CENTROPARIETAL, channel_names)
    plot_eeg_channels(cp_data, cp_names)

    # Compute and display statistics for both clusters
    print("Computing channel statistics for clusters...")
    fc_stats = compute_channel_statistics(fc_data, fc_names)
    cp_stats = compute_channel_statistics(cp_data, cp_names)

    print("Frontocentral Cluster Statistics:")
    print(fc_stats)
    print("Centroparietal Cluster Statistics:")
    print(cp_stats)

    # Save statistics to CSV files
    fc_stats.to_csv("frontocentral_statistics.csv", index=False)
    cp_stats.to_csv("centroparietal_statistics.csv", index=False)
    print("Cluster statistics saved to CSV files.")

if __name__ == "__main__":
    main()
