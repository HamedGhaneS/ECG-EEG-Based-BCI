import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
from datetime import datetime, timedelta

def load_feature_data(feature_dir: str) -> Dict:
    """
    Load extracted features and metadata.
    
    Args:
        feature_dir (str): Directory containing feature files
        
    Returns:
        dict: Dictionary containing features and metadata
    """
    print(f"Loading features from {feature_dir}...")
    
    # Load features
    feature_file = os.path.join(feature_dir, 'extracted_features.npz')
    feature_data = np.load(feature_file)
    
    # Load metadata
    metadata_file = os.path.join(feature_dir, 'features_metadata.csv')
    metadata = pd.read_csv(metadata_file)
    
    print(f"Successfully loaded data with {len(metadata['window_idx'].unique())} windows and {len(metadata['epoch_idx'].unique())} epochs")
    
    return {
        'features': feature_data['features'],
        'window_times': feature_data['window_times'],
        'channel_names': feature_data['channel_names'],
        'metadata': metadata
    }

def train_window_classifier(features: np.ndarray, labels: np.ndarray) -> Tuple[float, LinearDiscriminantAnalysis, np.ndarray, np.ndarray]:
    """
    Train classifier for a specific time window using leave-one-out cross validation.
    
    Args:
        features (np.ndarray): Feature array
        labels (np.ndarray): Labels
        
    Returns:
        tuple: (Az score, trained classifier, predicted probabilities, true labels)
    """
    # Initialize cross-validation
    loo = LeaveOneOut()
    y_pred = np.zeros_like(labels, dtype=float)
    y_pred_class = np.zeros_like(labels, dtype=int)
    
    # Train and predict
    total_folds = len(features)
    for i, (train_idx, test_idx) in enumerate(loo.split(features)):
        # Print progress every 50 folds
        if total_folds > 100 and i % 50 == 0 and i > 0:
            print(f"    Cross-validation progress: {i}/{total_folds} folds ({i/total_folds*100:.1f}%)")
            
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(features[train_idx], labels[train_idx])
        y_pred[test_idx] = clf.predict_proba(features[test_idx])[:, 1]
        y_pred_class[test_idx] = clf.predict(features[test_idx])
    
    # Calculate Az score
    az_score = roc_auc_score(labels, y_pred)
    
    # Train final classifier on all data
    final_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    final_clf.fit(features, labels)
    
    return az_score, final_clf, y_pred, y_pred_class

def compute_significance(features: np.ndarray, labels: np.ndarray, n_permutations: int = 100) -> float:
    """
    Compute significance level using permutation tests.
    
    Args:
        features (np.ndarray): Feature array
        labels (np.ndarray): Labels
        n_permutations (int): Number of permutations (reduced to 100 for speed)
        
    Returns:
        float: P-value
    """
    print(f"  Starting permutation tests (n={n_permutations})...")
    start_time = time.time()
    
    true_az, _, _, _ = train_window_classifier(features, labels)
    perm_scores = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        if i % 10 == 0 and i > 0:
            elapsed = time.time() - start_time
            time_per_perm = elapsed / i
            remaining = (n_permutations - i) * time_per_perm
            eta = str(timedelta(seconds=int(remaining)))
            print(f"    Permutation {i}/{n_permutations} ({i/n_permutations*100:.1f}%), ETA: {eta}")
            
        labels_perm = np.random.permutation(labels)
        perm_scores[i], _, _, _ = train_window_classifier(features, labels_perm)
    
    p_value = np.mean(perm_scores >= true_az)
    elapsed = time.time() - start_time
    print(f"  Permutation tests complete in {elapsed:.1f}s. P-value: {p_value:.4f}")
    return p_value

def plot_confusion_matrix(y_true, y_pred, classes, ax, title):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
def run_classification(feature_dir: str, output_dir: str, label_mapping: Dict[int, str] = None) -> None:
    """
    Run complete classification pipeline and save results.
    
    Args:
        feature_dir (str): Directory containing feature files
        output_dir (str): Directory to save results
        label_mapping (Dict[int, str]): Mapping from label values to class names
    """
    print("\n=== Starting Classification Pipeline ===")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    total_start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = load_feature_data(feature_dir)
    features = data['features']
    metadata = data['metadata']
    window_times = data['window_times']
    
    # Get unique windows
    unique_windows = metadata['window_idx'].unique()
    n_windows = len(unique_windows)
    
    # Set default label mapping if not provided
    if label_mapping is None:
        label_mapping = {0: 'Low', 1: 'High'}
    
    # Initialize results storage
    az_scores = np.zeros(n_windows)
    p_values = np.zeros(n_windows)
    accuracies = np.zeros(n_windows)
    classifiers = []
    all_predictions = []
    
    # Create progress tracking file
    progress_file = os.path.join(output_dir, 'progress.txt')
    with open(progress_file, 'w') as f:
        f.write(f"Classification started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total windows to process: {n_windows}\n\n")
    
    # Run classification for each window
    print(f"\nRunning classification for {n_windows} time windows...")
    for window_idx in range(n_windows):
        window_start = time.time()
        print(f"\nProcessing window {window_idx+1}/{n_windows} ({window_idx/n_windows*100:.1f}%)")
        print(f"Window time: {window_times[window_idx]:.3f}s")
        
        # Save progress to file (in case of crash)
        with open(progress_file, 'a') as f:
            f.write(f"Starting window {window_idx+1}/{n_windows} at {datetime.now().strftime('%H:%M:%S')}\n")
        
        # Get data for current window
        window_mask = metadata['window_idx'] == window_idx
        X = features[window_mask]
        y = metadata.loc[window_mask, 'label'].values
        
        # Report data shape and label distribution
        print(f"Data shape: {X.shape}")
        class_distribution = {label_mapping[label]: np.sum(y==label) for label in np.unique(y)}
        print(f"Label distribution: {class_distribution}")
        
        # Train classifier and get results
        print(f"Training classifier...")
        az_score, clf, y_pred, y_pred_class = train_window_classifier(X, y)
        az_scores[window_idx] = az_score
        accuracies[window_idx] = np.mean(y_pred_class == y)
        classifiers.append(clf)
        all_predictions.append((y, y_pred, y_pred_class))
        
        print(f"Window {window_idx+1} Az score: {az_score:.3f}, Accuracy: {accuracies[window_idx]:.3f}")
        
        # Compute significance if Az score is above chance
        if az_score > 0.5:
            p_values[window_idx] = compute_significance(X, y)
        else:
            p_values[window_idx] = 1.0
            print("  Skipping permutation tests (Az score <= 0.5)")
        
        # Save progress to file
        with open(progress_file, 'a') as f:
            f.write(f"  Completed in {time.time() - window_start:.1f}s\n")
            f.write(f"  Az score: {az_score:.3f}, p-value: {p_values[window_idx]:.4f}\n\n")
        
        # Save intermediate results periodically
        if (window_idx + 1) % 10 == 0 or window_idx == n_windows - 1:
            print(f"Saving intermediate results...")
            intermediate_results = {
                'az_scores': az_scores[:window_idx+1],
                'accuracies': accuracies[:window_idx+1],
                'p_values': p_values[:window_idx+1],
                'window_times': window_times[:window_idx+1]
            }
            np.savez(os.path.join(output_dir, 'classification_results_partial.npz'), **intermediate_results)
        
        window_elapsed = time.time() - window_start
        windows_left = n_windows - (window_idx + 1)
        est_time_left = windows_left * window_elapsed
        print(f"Window completed in {window_elapsed:.1f}s, estimated time remaining: {str(timedelta(seconds=int(est_time_left)))}")
    
    # Save results to dataframe for easier analysis
    results_df = pd.DataFrame({
        'window_time': window_times,
        'az_score': az_scores,
        'accuracy': accuracies,
        'p_value': p_values,
        'significant': p_values < 0.05
    })
    
    # Save results
    results = {
        'az_scores': az_scores,
        'accuracies': accuracies,
        'p_values': p_values,
        'window_times': window_times
    }
    np.savez(os.path.join(output_dir, 'classification_results.npz'), **results)
    results_df.to_csv(os.path.join(output_dir, 'classification_results.csv'), index=False)
    
    # Find best window
    best_window_idx = np.argmax(az_scores)
    best_time = window_times[best_window_idx]
    best_az = az_scores[best_window_idx]
    best_accuracy = accuracies[best_window_idx]
    best_p = p_values[best_window_idx]
    
    print(f"\nCreating visualizations...")
    
    # Get classification details for best window
    best_window_mask = metadata['window_idx'] == best_window_idx
    y_true = metadata.loc[best_window_mask, 'label'].values
    _, _, y_pred_prob, y_pred_class = train_window_classifier(
        features[best_window_mask], y_true)
    
    # Create plots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Az score over time
    axs[0, 0].plot(window_times, az_scores, 'b-', linewidth=2)
    axs[0, 0].axhline(y=0.5, color='r', linestyle='--', label='Chance level')
    axs[0, 0].fill_between(window_times, az_scores, 0.5, 
                     where=(p_values < 0.05), color='gray', alpha=0.3,
                     label='p < 0.05')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Az Score')
    axs[0, 0].set_title('Classification Performance Across Time')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Accuracy over time
    axs[0, 1].plot(window_times, accuracies, 'g-', linewidth=2)
    axs[0, 1].axhline(y=0.5, color='r', linestyle='--', label='Chance level')
    axs[0, 1].fill_between(window_times, accuracies, 0.5, 
                     where=(p_values < 0.05), color='gray', alpha=0.3,
                     label='p < 0.05')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].set_title('Classification Accuracy Across Time')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # P-values
    axs[1, 0].plot(window_times, p_values, 'r-', linewidth=2)
    axs[1, 0].axhline(y=0.05, color='k', linestyle='--', label='p=0.05')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('P-value')
    axs[1, 0].set_title('Statistical Significance Across Time')
    axs[1, 0].set_yscale('log')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Confusion matrix
    class_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    plot_confusion_matrix(y_true, y_pred_class, class_names, 
                          axs[1, 1], f'Confusion Matrix at {best_time:.3f}s')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_results.png'))
    
    # Generate report
    report_text = classification_report(y_true, y_pred_class, 
                                       target_names=class_names)
    
    # Save text report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("CLASSIFICATION RESULTS SUMMARY\n")
        f.write("=============================\n\n")
        
        f.write(f"Time window analyzed: {window_times[0]:.3f}s to {window_times[-1]:.3f}s\n")
        f.write(f"Total windows: {n_windows}\n")
        f.write(f"Number of samples: {len(features)}\n")
        f.write(f"Number of epochs: {len(metadata['epoch_idx'].unique())}\n\n")
        
        f.write("BEST PERFORMANCE\n")
        f.write(f"Time: {best_time:.3f}s\n")
        f.write(f"Az score: {best_az:.3f}\n")
        f.write(f"Accuracy: {best_accuracy:.3f}\n")
        f.write(f"P-value: {best_p:.6f}\n\n")
        
        f.write("NUMBER OF SIGNIFICANT WINDOWS\n")
        f.write(f"p < 0.05: {np.sum(p_values < 0.05)}\n")
        f.write(f"p < 0.01: {np.sum(p_values < 0.01)}\n")
        f.write(f"p < 0.001: {np.sum(p_values < 0.001)}\n\n")
        
        f.write("CLASSIFICATION REPORT FOR BEST WINDOW\n")
        f.write(report_text)
        
        f.write("\n\nPROCESSING INFORMATION\n")
        f.write(f"Total processing time: {time.time() - total_start_time:.1f} seconds\n")
        f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Final output
    print("\nClassification Results:")
    print(f"Number of time windows: {n_windows}")
    print(f"Time window range: {window_times[0]:.3f}s to {window_times[-1]:.3f}s")
    print(f"Best Az score: {best_az:.3f} at {best_time:.3f}s")
    print(f"Best accuracy: {best_accuracy:.3f}")
    print(f"Number of significant windows (p < 0.05): {np.sum(p_values < 0.05)}")
    print(f"Results saved to: {output_dir}")
    print(f"Total processing time: {(time.time() - total_start_time)/60:.1f} minutes")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # Base paths
    base_feature_dir = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.02.24\ML Classification\Extracted_Features"
    base_output_dir = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.02.24\ML Classification\Classification_Results"
    
    # Ask the user for classification type
    print("\n=== Classification Configuration ===")
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
    
    # Define paths and parameters based on classification type
    if classification_type == 1:
        subfolder = "Original_HighLow"
        label_mapping = {0: 'Low', 1: 'High'}
        description = "Original High vs Low"
    elif classification_type == 2:
        subfolder = "Quantile_33"
        label_mapping = {0: 'Low Surprising', 1: 'High Surprising'}
        description = "33% Quantile (High Surprising vs Low Surprising)"
    elif classification_type == 3:
        subfolder = "Quantile_25"
        label_mapping = {0: 'Low Surprising', 1: 'High Surprising'}
        description = "25% Quantile (High Surprising vs Low Surprising)"
    
    # Configure feature and output directories
    feature_dir = os.path.join(base_feature_dir, subfolder)
    output_dir = os.path.join(base_output_dir, subfolder)
    
    print(f"\nUsing classification type: {description}")
    print(f"Feature directory: {feature_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if feature directory exists
    if not os.path.exists(feature_dir):
        print(f"\nError: Feature directory {feature_dir} does not exist.")
        print("Please run feature extraction for this classification type first.")
        exit(1)
    
    # Run classification
    run_classification(feature_dir, output_dir, label_mapping)
