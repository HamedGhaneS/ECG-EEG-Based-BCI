import mne
import numpy as np
import pandas as pd
from typing import Dict
import os

def extract_windowed_features(epochs_path: str, base_output_dir: str, classification_type: int = 1, 
                              window_size: int = 60, step_size: int = 10) -> None:
    """
    Extract features using sliding windows from EEG epochs.
    Time window: 0.000 to 0.600 seconds
    With 60ms windows, effective time range will be 0.030 to 0.570 seconds
    
    Args:
        epochs_path (str): Path to the epochs file
        base_output_dir (str): Base directory to save the extracted features
        classification_type (int): Type of classification to use:
            1: Original AbsPE_Level (high vs low)
            2: 33% quantile (high_surprising vs low_surprising)
            3: 25% quantile (high_surprising vs low_surprising)
        window_size (int): Size of sliding window in ms
        step_size (int): Step size for sliding window in ms
    """
    print("\n=== Starting Feature Extraction ===")
    
    # Define subdirectory based on classification type
    if classification_type == 1:
        subfolder = "Original_HighLow"
        label_column = 'AbsPE_Level'
        high_label = 'high'
        low_label = 'low'
        description = "Original High vs Low"
    elif classification_type == 2:
        subfolder = "Quantile_33"
        label_column = 'AbsPE_Surprise_q33'
        high_label = 'high_surprising'
        low_label = 'low_surprising'
        description = "33% Quantile (High Surprising vs Low Surprising)"
    elif classification_type == 3:
        subfolder = "Quantile_25"
        label_column = 'AbsPE_Surprise_q25'
        high_label = 'high_surprising'
        low_label = 'low_surprising'
        description = "25% Quantile (High Surprising vs Low Surprising)"
    else:
        raise ValueError("Invalid classification type. Must be 1, 2, or 3.")
    
    # Create output directory
    output_dir = os.path.join(base_output_dir, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nUsing classification type: {description}")
    print(f"Output directory: {output_dir}")
    
    # Load epochs
    print("\nLoading epochs...")
    epochs = mne.read_epochs(epochs_path)
    
    # Crop epochs to 0-0.6s
    epochs_cropped = epochs.copy().crop(tmin=0, tmax=0.6)
    print(f"Loaded {len(epochs_cropped)} epochs")
    
    # Filter epochs based on classification type
    if classification_type > 1:  # For quantile-based classifications
        # Keep only high_surprising and low_surprising
        mask = ((epochs_cropped.metadata[label_column] == high_label) | 
                (epochs_cropped.metadata[label_column] == low_label))
        epochs_filtered = epochs_cropped[mask]
        print(f"Filtered to {len(epochs_filtered)} epochs ({len(epochs_filtered)/len(epochs_cropped)*100:.1f}% of original)")
    else:
        epochs_filtered = epochs_cropped
    
    # Get labels based on the selected column
    labels = np.where(epochs_filtered.metadata[label_column].str.lower() == high_label, 1, 0)
    print(f"\nLabel distribution:")
    print(f"{high_label}: {np.sum(labels == 1)}")
    print(f"{low_label}: {np.sum(labels == 0)}")
    
    # Parameters
    sfreq = 5000  # Sampling rate in Hz
    window_samples = int(window_size * sfreq / 1000)  # 60ms = 300 samples
    step_samples = int(step_size * sfreq / 1000)      # 10ms = 50 samples
    
    # Calculate time points
    times = epochs_filtered.times  # Get actual time points from epochs
    start_time = times[0]  # Should be 0.0
    end_time = times[-1]   # Should be 0.6
    
    # Calculate number of windows that will fit in the time range
    total_samples = len(times)
    n_windows = (total_samples - window_samples) // step_samples + 1
    
    print(f"\nProcessing parameters:")
    print(f"Time window: {start_time:.3f}s to {end_time:.3f}s")
    print(f"Window size: {window_size}ms ({window_samples} samples)")
    print(f"Step size: {step_size}ms ({step_samples} samples)")
    print(f"Number of windows per epoch: {n_windows}")
    print(f"Number of channels: {len(epochs_filtered.ch_names)}")
    
    # Get all data
    all_data = epochs_filtered.get_data()  # Shape: (n_epochs, n_channels, n_timepoints)
    
    # Initialize feature array to store averaged data
    features = np.zeros((n_windows * len(epochs_filtered), len(epochs_filtered.ch_names)))
    
    # Initialize arrays for metadata
    all_labels = np.zeros(n_windows * len(epochs_filtered))
    window_times = np.zeros(n_windows)
    epoch_indices = np.zeros(n_windows * len(epochs_filtered))
    window_indices = np.zeros(n_windows * len(epochs_filtered))
    
    # Extract windows and average across time
    print("\nProcessing time windows...")
    for i in range(n_windows):
        if i % 10 == 0:
            print(f"Processing window {i+1}/{n_windows}")
            
        start_sample = i * step_samples
        end_sample = start_sample + window_samples
        
        # Use actual time points from epochs
        window_times[i] = times[start_sample + window_samples//2]
        
        # Average across time dimension for each window
        window_data = np.mean(all_data[:, :, start_sample:end_sample], axis=2)
        
        # Store data and metadata
        start_idx = i * len(epochs_filtered)
        end_idx = (i + 1) * len(epochs_filtered)
        features[start_idx:end_idx] = window_data
        all_labels[start_idx:end_idx] = labels
        epoch_indices[start_idx:end_idx] = np.arange(len(epochs_filtered))
        window_indices[start_idx:end_idx] = i
    
    # Create metadata DataFrame
    metadata = pd.DataFrame({
        'label': all_labels,
        'epoch_idx': epoch_indices,
        'window_idx': window_indices,
        'window_time': np.repeat(window_times, len(epochs_filtered))
    })
    
    # Save features and metadata
    feature_file = os.path.join(output_dir, 'extracted_features.npz')
    np.savez(feature_file,
             features=features,
             window_times=window_times,
             channel_names=epochs_filtered.ch_names)
    
    metadata_file = os.path.join(output_dir, 'features_metadata.csv')
    metadata.to_csv(metadata_file, index=False)
    
    # Save classification details for reference
    with open(os.path.join(output_dir, 'classification_info.txt'), 'w') as f:
        f.write(f"Classification Type: {description}\n")
        f.write(f"Label Column: {label_column}\n")
        f.write(f"High Label: {high_label}\n")
        f.write(f"Low Label: {low_label}\n")
        f.write(f"Number of Epochs: {len(epochs_filtered)}\n")
        f.write(f"Label Distribution:\n")
        f.write(f"  {high_label}: {np.sum(labels == 1)}\n")
        f.write(f"  {low_label}: {np.sum(labels == 0)}\n")
    
    print("\nFeature extraction complete!")
    print(f"Features shape: {features.shape}")
    print(f"Window times range: {window_times[0]:.3f}s to {window_times[-1]:.3f}s")
    print(f"Saved features to: {feature_file}")
    print(f"Saved metadata to: {metadata_file}")
    print(f"Classification info saved to: {os.path.join(output_dir, 'classification_info.txt')}")

if __name__ == "__main__":
    # Paths
    epochs_path = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.02.24\ML Classification\New Epochs(Surprizing AbsPEs)\cardiac_epochs_cleaned_reref\R1_Locked-prepro_epochs-epo.fif"
    base_output_dir = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.02.24\ML Classification\Extracted_Features"
    
    # Ask the user for classification type
    print("\n=== Feature Extraction Configuration ===")
    print("Select classification type:")
    print("1: Original High vs Low (as before)")
    print("2: 33% Quantile (High Surprising vs Low Surprising)")
    print("3: 25% Quantile (High Surprising vs Low Surprising)")
    
    while True:
        try:
            classification_type = int(input("Enter your choice (1-3): "))
            if classification_type in [1, 2, 3]:
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Run feature extraction with the selected classification type
    extract_windowed_features(epochs_path, base_output_dir, classification_type)
