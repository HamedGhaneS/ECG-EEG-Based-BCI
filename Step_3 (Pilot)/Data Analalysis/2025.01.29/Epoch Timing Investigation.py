import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

def analyze_excel_timing(timing_file):
    """Analyze timing patterns in the Excel data."""
    print("\n=== Excel Timing Analysis ===")
    timing_data = pd.read_excel(timing_file)
    
    # Analyze absolute PE conditions
    print("\nAbsolute PE Analysis:")
    high_pe = timing_data[timing_data['abs_pe_level'].str.lower() == 'high']
    low_pe = timing_data[timing_data['abs_pe_level'].str.lower() == 'low']
    
    timing_columns = ['outcome_time', 'r1_time', 'r2_time', 'r3_time']
    
    for col in timing_columns:
        print(f"\n{col} Statistics:")
        print(f"High PE: mean={high_pe[col].mean():.3f}s, std={high_pe[col].std():.3f}s")
        print(f"Low PE: mean={low_pe[col].mean():.3f}s, std={low_pe[col].std():.3f}s")
        print(f"Difference (High - Low): {(high_pe[col].mean() - low_pe[col].mean())*1000:.2f}ms")
    
    # Analyze signed PE conditions
    print("\nSigned PE Analysis:")
    pos_pe = timing_data[timing_data['pe_value'] > 0]
    neg_pe = timing_data[timing_data['pe_value'] < 0]
    
    for col in timing_columns:
        print(f"\n{col} Statistics:")
        print(f"Positive PE: mean={pos_pe[col].mean():.3f}s, std={pos_pe[col].std():.3f}s")
        print(f"Negative PE: mean={neg_pe[col].mean():.3f}s, std={neg_pe[col].std():.3f}s")
        print(f"Difference (Pos - Neg): {(pos_pe[col].mean() - neg_pe[col].mean())*1000:.2f}ms")
    
    return timing_data

def analyze_epoch_timing(epoch_file):
    """Analyze timing in epoch files."""
    print(f"\n=== Analyzing {os.path.basename(epoch_file)} ===")
    epochs = mne.read_epochs(epoch_file, preload=True)
    
    # Analyze event timing
    events = epochs.events
    event_id = epochs.event_id
    sfreq = epochs.info['sfreq']
    
    # Get timing relative to first event
    first_event_time = events[0, 0] / sfreq
    
    print("\nEvent Timing Analysis:")
    for condition, event_num in event_id.items():
        cond_events = events[events[:, 2] == event_num]
        event_times = cond_events[:, 0] / sfreq - first_event_time
        
        print(f"\nCondition: {condition}")
        print(f"Number of events: {len(cond_events)}")
        print(f"Mean time from first event: {event_times.mean():.3f}s")
        print(f"Time range: {event_times.min():.3f}s to {event_times.max():.3f}s")
        print(f"Standard deviation: {event_times.std():.3f}s")
    
    return epochs

def plot_event_distribution(epoch_file, output_dir):
    """Plot event time distributions."""
    epochs = mne.read_epochs(epoch_file, preload=True)
    events = epochs.events
    event_id = epochs.event_id
    sfreq = epochs.info['sfreq']
    
    plt.figure(figsize=(12, 6))
    
    colors = ['red', 'blue']
    first_event_time = events[0, 0] / sfreq
    
    for (condition, event_num), color in zip(event_id.items(), colors):
        cond_events = events[events[:, 2] == event_num]
        event_times = cond_events[:, 0] / sfreq - first_event_time
        
        plt.hist(event_times, bins=30, alpha=0.5, color=color, label=condition)
    
    plt.title(f'Event Time Distribution - {os.path.basename(epoch_file)}')
    plt.xlabel('Time from first event (s)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = os.path.join(output_dir, f'event_distribution_{os.path.basename(epoch_file)[:-4]}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set up paths
    base_dir = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.28\Pilot Data Analysis\Enhanced Epoching"
    
    while True:
        participant = input("Enter participant name (Elmira/Harry) or 'q' to quit: ").strip()
        
        if participant.lower() == 'q':
            break
            
        if participant not in ["Elmira", "Harry"]:
            print("Invalid participant name. Please try again.")
            continue
        
        # Set up participant-specific paths
        participant_dir = os.path.join(base_dir, participant)
        if participant == "Elmira":
            timing_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.22\Pilot Data Analysis\Elmira\Trial Details\timing_analysis\trial_timing_data_cleaned.xlsx"
        else:
            timing_file = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.20\Pilot Data Analysis\timing_analysis\trial_timing_data_cleaned.xlsx"
            
        # Create output directory for plots
        output_dir = os.path.join(participant_dir, 'Timing_Analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # Analyze Excel timing data
        print(f"\nAnalyzing timing data for {participant}")
        timing_data = analyze_excel_timing(timing_file)
        
        # Analyze epoch files
        conditions = ['abs_pe_epochs', 'pe_sign_epochs']
        
        for condition in conditions:
            condition_dir = os.path.join(participant_dir, condition)
            if not os.path.exists(condition_dir):
                print(f"Warning: {condition} directory not found")
                continue
            
            # Analyze each epoch file
            for epoch_file in Path(condition_dir).glob('*-prepro_epochs-epo.fif'):
                try:
                    epochs = analyze_epoch_timing(str(epoch_file))
                    plot_event_distribution(str(epoch_file), output_dir)
                except Exception as e:
                    print(f"Error processing {epoch_file.name}: {str(e)}")

if __name__ == "__main__":
    main()