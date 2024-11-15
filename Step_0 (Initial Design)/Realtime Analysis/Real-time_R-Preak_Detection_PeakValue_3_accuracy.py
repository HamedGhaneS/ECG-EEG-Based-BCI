"""
Real-time ECG Processing and R-Peak Detection with Enhanced Accuracy Reporting
Author: Hamed Ghane
Date: 2024-11-15

SUMMARY:
This script performs real-time R-peak detection on ECG data received through LSL streaming, implementing a modified Pan-Tompkins algorithm with adaptive thresholding for peak detection. 
The system continuously monitors detection accuracy and processing delays, providing immediate feedback on each detected peak while storing comprehensive timing and amplitude information. 
Upon completion, it performs a thorough post-processing validation comparing real-time detections against true peaks, generating detailed accuracy metrics, visualizations, and a complete performance report.


WORKFLOW AND FUNCTIONALITY:

1. Data Acquisition:
   - Connects to LSL stream providing ECG data at 1000 Hz
   - Validates stream connection and parameters
   - Initializes data buffers and processing parameters
   - Checks time domain synchronization between LSL and system time

2. Initial Training Phase:
   - Collects first 10 seconds of ECG data
   - Applies bandpass filtering (0.5-40 Hz)
   - Calculates initial QRS detection thresholds
   - Establishes baseline for peak detection
   - Determines initial systolic duration

3. Real-time Processing:
   - Implements modified Pan-Tompkins algorithm for R-peak detection
   - Uses 100ms sliding window for peak analysis
   - Processes each sample for potential R-peaks
   - Measures three critical timings:
     a) Sample arrival time (LSL timestamp)
     b) Peak detection time (local_clock)
     c) Processing delay (detection - arrival time)

4. Detection Parameters:
   - Adaptive threshold updating based on signal characteristics
   - Maintains minimum 200ms between consecutive peaks
   - Tracks cardiac phase (systole/diastole)
   - Updates QRS parameters based on detected peaks

5. Real-time Performance Metrics:
   - Tracks instantaneous processing delay
   - Calculates running average of detection delay
   - Monitors detection consistency
   - Reports timing statistics for each detected peak

6. Post-Processing Validation:
   - Analyzes complete recorded ECG signal
   - Detects "true" R-peaks using prominence-based criteria
   - Compares real-time detections with post-processed peaks
   - Validates detection accuracy and timing precision

7. Accuracy Metrics:
   - Peak Detection Statistics:
     * Total true peaks vs. detected peaks
     * True positives, false positives, false negatives
   - Performance Metrics:
     * Sensitivity (Recall) = TP/(TP + FN)
     * Precision = TP/(TP + FP)
     * F1 Score = 2*(Precision*Recall)/(Precision + Recall)
   - Timing Analysis:
     * Mean detection delay
     * Standard deviation of delay
     * Maximum and minimum delays
     * Detection error distribution

8. Data Export and Visualization:
   - Saves processed data:
     * Full ECG signal with timestamps
     * Detection results with timing information
     * Comprehensive accuracy metrics
   - Generates visualizations:
     * ECG signal with marked peaks
     * True vs. detected peak comparison
     * Timing error distribution
     * Performance metrics summary

9. Quality Assessment:
   - Evaluates overall detection quality:
     * Excellent: F1 Score ≥ 95%
     * Very Good: F1 Score ≥ 90%
     * Good: F1 Score ≥ 85%
     * Fair: F1 Score ≥ 80%
     * Needs Improvement: F1 Score < 80%

OUTPUTS:
1. Real-time Reporting:
   - Peak detection events
   - Current and average processing delays
   - Cardiac phase information

2. Final Analysis:
   - Comprehensive accuracy report
   - Detection quality assessment
   - Timing performance statistics
   - Visual analysis plots

3. Data Files:
   - {timestamp}_full_stream.csv: Complete ECG recording
   - {timestamp}_detections.csv: Peak detection results
   - {timestamp}_accuracy_metrics.csv: Performance metrics
   - {timestamp}_analysis.png: Visualization plots

USAGE:
1. Ensure LSL stream is running
2. Run script
3. Monitor real-time detection
4. Press 'q' to stop and view analysis
"""


import numpy as np
from pylsl import StreamInlet, resolve_stream, local_clock
from scipy.signal import butter, sosfilt, sosfiltfilt, find_peaks
import time
from collections import deque
from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
import keyboard
import warnings
warnings.filterwarnings('ignore')

class RealtimeECGProcessor:
    def __init__(self, sampling_rate=1000, training_duration=10):
        """Initialize the real-time ECG processor"""
        self.sampling_rate = sampling_rate
        self.training_duration = training_duration
        
        # Add the time domain difference check
        self.time_domain_diff = local_clock() - time.time()
        print(f"Time domain difference: {self.time_domain_diff:.6f} seconds")
        
        # Buffer parameters
        self.r_peak_window = int(0.100 * sampling_rate)  # 100ms window
        
        # Processing parameters
        self.qrs_peak_threshold = 0
        self.noise_threshold = 0
        self.current_phase = "diastole"
        self.systole_duration = 0.3  # Will be adjusted during training
        
        # LSL stream validation
        self.stream_name = 'ECGSimulation'
        self.stream_type = 'ECG'
        
        # Initialize all components
        self.setup_stream()
        self.setup_buffers()
        self.setup_filters()
        self.setup_data_export()
        
        # Additional attributes for validation
        self.realtime_peaks = []
        self.realtime_peak_values = []
        self.realtime_detection_times = []

    def setup_stream(self):
        """Setup and validate LSL stream connection"""
        print("Looking for ECG stream...")
        streams = resolve_stream('type', self.stream_type)
        
        for stream in streams:
            if stream.name() == self.stream_name:
                self.stream_info = stream
                self.inlet = StreamInlet(stream)
                print(f"Found matching stream: {stream.name()}")
                return
                
        raise RuntimeError("Could not find matching LSL stream")
    
    def setup_buffers(self):
        """Initialize all data buffers"""
        self.training_buffer = []
        self.is_training = True
        self.signal_buffer = deque(maxlen=int(2 * self.sampling_rate))
        self.r_peaks = deque(maxlen=10)
        self.r_peak_times = deque(maxlen=10)
        self.detection_times = deque(maxlen=10)
        self.detection_delays = deque(maxlen=1000)
        self.total_detections = 0
        self.cumulative_delay = 0
    
    def setup_filters(self):
        """Setup Butterworth bandpass filter (0.5-40 Hz)"""
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        self.sos = butter(4, [low, high], btype='band', output='sos')
    
    def setup_data_export(self):
        """Setup data export structures and files"""
        self.export_data = {
            'true_peak_times': [],
            'detection_times': [],
            'detection_delays': [],
            'rr_intervals': [],
            'cardiac_phases': [],
            'peak_values': []
        }
        
        self.full_ecg_data = []
        self.export_dir = 'ecg_analysis_results'
        os.makedirs(self.export_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.export_filename = f'ecg_analysis_{timestamp}'

    def train_detector(self):
        """Train the R-peak detector using initial data"""
        print("Training R-peak detector...")
        
        signal = np.array([x[0] for x in self.training_buffer])
        times = np.array([x[1] for x in self.training_buffer])
        
        filtered_signal = sosfiltfilt(self.sos, signal)
        diff_signal = np.diff(filtered_signal)
        squared_signal = diff_signal ** 2
        
        window_size = int(0.150 * self.sampling_rate)
        ma_signal = np.convolve(squared_signal, np.ones(window_size)/window_size, mode='same')
        
        self.qrs_peak_threshold = np.mean(ma_signal) + 2 * np.std(ma_signal)
        self.noise_threshold = np.mean(ma_signal) + np.std(ma_signal)
        
        # Initial R-peak detection
        peaks = self.find_r_peaks(ma_signal)
        
        if len(peaks) >= 2:
            rr_intervals = np.diff([times[p] for p in peaks])
            mean_rr = np.mean(rr_intervals)
            self.systole_duration = mean_rr * 0.3
        
        print("Training complete. Parameters configured.")
        self.is_training = False

    def find_r_peaks(self, signal):
        """Simple R-peak detection for training phase"""
        peaks = []
        for i in range(self.r_peak_window, len(signal) - self.r_peak_window):
            if signal[i] > self.qrs_peak_threshold:
                if signal[i] == max(signal[i-self.r_peak_window:i+self.r_peak_window]):
                    peaks.append(i)
        return peaks

    def process_sample(self, sample, timestamp):
        """Process a new ECG sample in real-time"""
        if self.is_training:
            self.training_buffer.append((sample, timestamp))
            if len(self.training_buffer) >= self.training_duration * self.sampling_rate:
                self.train_detector()
            return False, "unknown", None
        
        self.full_ecg_data.append((timestamp, sample))
        self.signal_buffer.append((sample, timestamp))
        
        if len(self.signal_buffer) < self.r_peak_window * 2:
            return False, self.current_phase, None
        
        window_data = np.array([s[0] for s in list(self.signal_buffer)[-self.r_peak_window*2:]])
        window_times = np.array([s[1] for s in list(self.signal_buffer)[-self.r_peak_window*2:]])
        
        filtered_window = sosfilt(self.sos, window_data)
        diff_signal = np.diff(filtered_window)
        squared_signal = diff_signal ** 2
        
        is_r_peak = False
        delays = None
        
        if squared_signal[-1] > self.qrs_peak_threshold:
            if squared_signal[-1] == max(squared_signal[-self.r_peak_window:]):
                if not self.r_peak_times or (timestamp - self.r_peak_times[-1]) > 0.2:
                    is_r_peak = True
                    detection_time = local_clock()
                    peak_value = window_data[-1]
                    
                    print(f"""
                    Timestamps (LSL time domain):
                    Sample arrival: {timestamp:.6f}
                    Peak Value: {peak_value:.6f}
                    Detection:     {detection_time:.6f}
                    Delay:        {(detection_time - timestamp)*1000:.3f} ms
                    """)
                    
                    # Store peak information
                    self.r_peak_times.append(timestamp)
                    self.detection_times.append(detection_time)
                    self.realtime_peaks.append(timestamp)
                    self.realtime_peak_values.append(peak_value)
                    self.realtime_detection_times.append(detection_time)
                    
                    # Calculate delays
                    current_delay = detection_time - timestamp
                    self.detection_delays.append(current_delay)
                    self.total_detections += 1
                    self.cumulative_delay += current_delay
                    
                    delays = (current_delay, self.cumulative_delay / self.total_detections)
                    
                    # Store export data
                    self.export_data['true_peak_times'].append(timestamp)
                    self.export_data['detection_times'].append(detection_time)
                    self.export_data['detection_delays'].append(current_delay)
                    self.export_data['cardiac_phases'].append(self.current_phase)
                    self.export_data['peak_values'].append(peak_value)
                    
                    if len(self.export_data['true_peak_times']) > 1:
                        self.export_data['rr_intervals'].append(
                            timestamp - self.export_data['true_peak_times'][-2])
                    
                    self.qrs_peak_threshold = 0.875 * self.qrs_peak_threshold + 0.125 * squared_signal[-1]
                    self.current_phase = "systole"
        else:
            if self.r_peak_times and timestamp - self.r_peak_times[-1] > self.systole_duration:
                self.current_phase = "diastole"
        
        return is_r_peak, self.current_phase, delays

    def detect_true_peaks(self, signal, times, min_distance=200):
        """
        Optimized true R-peaks detection with balanced sensitivity
        
        Args:
            signal: Raw ECG signal
            times: Corresponding timestamps
            min_distance: Minimum samples between peaks (default: 200ms at 1000Hz)
        """
        # Apply bandpass filter to clean signal
        filtered_signal = sosfiltfilt(self.sos, signal)
        
        # Calculate signal statistics
        signal_std = np.std(filtered_signal)
        signal_mean = np.mean(filtered_signal)
        
        # Initial peak detection with moderate criteria
        peaks, properties = find_peaks(
            filtered_signal,
            distance=min_distance,
            prominence=(signal_std * 0.35, None),  # Moderate prominence requirement
            height=(signal_mean + signal_std * 0.25, None),  # Moderate height threshold
            width=(None, 120)  # Moderate width constraint
        )
        
        # Calculate adaptive threshold using moderate percentile
        peak_heights = filtered_signal[peaks]
        height_threshold = np.percentile(peak_heights, 30)  # Balanced percentile
        
        # Validate peaks with balanced criteria
        true_peaks = []
        for i, peak in enumerate(peaks):
            peak_height = filtered_signal[peak]
            peak_prominence = properties['prominences'][i]
            
            # Use both conditions for better balance
            is_valid = (
                peak_height > height_threshold and  # Must meet height requirement
                peak_prominence > signal_std * 0.35  # Must have sufficient prominence
            )
            
            if is_valid:
                # Moderate window for local maximum check
                window_start = max(0, peak - 40)
                window_end = min(len(filtered_signal), peak + 40)
                local_max = np.argmax(filtered_signal[window_start:window_end]) + window_start
                
                if local_max not in true_peaks:
                    true_peaks.append(local_max)
        
        true_peaks.sort()
        
        # Get corresponding times and values
        true_peak_times = times[true_peaks]
        true_peak_values = signal[true_peaks]
        
        # Print detection statistics
        print(f"\nTrue Peak Detection Statistics:")
        print(f"Total peaks found: {len(true_peaks)}")
        print(f"Signal statistics:")
        print(f"  Mean: {signal_mean:.6f}")
        print(f"  Std: {signal_std:.6f}")
        print(f"  Height threshold: {height_threshold:.6f}")
        
        return true_peaks, true_peak_times, true_peak_values

    def calculate_accuracy_metrics(self, true_peaks, detected_peaks, window=0.05):
        """Calculate comprehensive accuracy metrics"""
        total_true = len(true_peaks)
        total_detected = len(detected_peaks)
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        matched_delays = []
        
        for true_time in true_peaks:
            matches = [d for d in detected_peaks if abs(d - true_time) <= window]
            if matches:
                true_positives += 1
                matched_delays.append(min(abs(d - true_time) for d in matches))
            else:
                false_negatives += 1
        
        false_positives = total_detected - true_positives
        
        sensitivity = true_positives / total_true if total_true > 0 else 0
        precision = true_positives / total_detected if total_detected > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        return {
            'Total True Peaks': total_true,
            'Total Detected Peaks': total_detected,
            'True Positives': true_positives,
            'False Positives': false_positives,
            'False Negatives': false_negatives,
            'Sensitivity (Recall)': sensitivity * 100,
            'Precision': precision * 100,
            'F1 Score': f1_score * 100,
            'Mean Detection Delay': np.mean(matched_delays) * 1000 if matched_delays else 0,
            'Std Detection Delay': np.std(matched_delays) * 1000 if matched_delays else 0,
            'Max Detection Delay': max(matched_delays) * 1000 if matched_delays else 0,
            'Min Detection Delay': min(matched_delays) * 1000 if matched_delays else 0
        }

    def print_accuracy_report(self, metrics):
        """Print a formatted accuracy report"""
        print("\n" + "="*50)
        print("             ACCURACY REPORT                ")
        print("="*50)
        
        print("\nPeak Detection Statistics:")
        print(f"Total True Peaks:     {metrics['Total True Peaks']}")
        print(f"Total Detected Peaks: {metrics['Total Detected Peaks']}")
        print(f"True Positives:       {metrics['True Positives']}")
        print(f"False Positives:      {metrics['False Positives']}")
        print(f"False Negatives:      {metrics['False Negatives']}")
        
        print("\nPerformance Metrics:")
        print(f"Sensitivity (Recall): {metrics['Sensitivity (Recall)']:.2f}%")
        print(f"Precision:           {metrics['Precision']:.2f}%")
        print(f"F1 Score:           {metrics['F1 Score']:.2f}%")
        
        print("\nTiming Analysis:")
        print(f"Mean Detection Delay: {metrics['Mean Detection Delay']:.2f} ms")
        print(f"Std Detection Delay:  {metrics['Std Detection Delay']:.2f} ms")
        print(f"Max Detection Delay:  {metrics['Max Detection Delay']:.2f} ms")
        print(f"Min Detection Delay:  {metrics['Min Detection Delay']:.2f} ms")
        
        if metrics['Total True Peaks'] > 0:
            detection_rate = (metrics['True Positives'] / metrics['Total True Peaks']) * 100
            print(f"\nOverall Detection Rate: {detection_rate:.2f}%")
        
        print("\nQuality Assessment:")
        if metrics['F1 Score'] >= 95:
            print("Excellent Detection Quality (F1 Score ≥ 95%)")
        elif metrics['F1 Score'] >= 90:
            print("Very Good Detection Quality (F1 Score ≥ 90%)")
        elif metrics['F1 Score'] >= 85:
            print("Good Detection Quality (F1 Score ≥ 85%)")
        elif metrics['F1 Score'] >= 80:
            print("Fair Detection Quality (F1 Score ≥ 80%)")
        else:
            print("Detection Quality Needs Improvement (F1 Score < 80%)")
        
        print("="*50)

    def export_results(self):
        """Export analysis results with validation"""
        ecg_data = np.array([x[1] for x in self.full_ecg_data])
        timestamps = np.array([x[0] for x in self.full_ecg_data])
        
        # Detect true peaks
        true_peaks, true_peak_times, true_peak_values = self.detect_true_peaks(ecg_data, timestamps)
        
        # Calculate accuracy metrics
        accuracy_metrics = self.calculate_accuracy_metrics(
            true_peak_times,
            np.array(self.export_data['true_peak_times'])
        )
        
        # Print accuracy report
        self.print_accuracy_report(accuracy_metrics)
        
        # Save full ECG data
        ecg_df = pd.DataFrame(self.full_ecg_data, columns=['Timestamp', 'ECG_Signal'])
        ecg_df.to_csv(f'{self.export_dir}/{self.export_filename}_full_stream.csv', index=False)
        
        # Save detection results - separate true peaks and detected peaks
        true_peaks_df = pd.DataFrame({
            'Time': true_peak_times,
            'Value': true_peak_values,
            'Type': ['True'] * len(true_peak_times)
        })
        
        detected_peaks_df = pd.DataFrame({
            'Time': self.export_data['true_peak_times'],
            'Value': self.export_data['peak_values'],
            'Type': ['Detected'] * len(self.export_data['true_peak_times']),
            'Detection_Time': self.export_data['detection_times'],
            'Detection_Delay_ms': [d * 1000 for d in self.export_data['detection_delays']],
            'Cardiac_Phase': self.export_data['cardiac_phases']
        })
        
        # Save both DataFrames
        true_peaks_df.to_csv(f'{self.export_dir}/{self.export_filename}_true_peaks.csv', index=False)
        detected_peaks_df.to_csv(f'{self.export_dir}/{self.export_filename}_detected_peaks.csv', index=False)
        
        # Save accuracy metrics
        metrics_df = pd.DataFrame([accuracy_metrics])
        metrics_df.to_csv(f'{self.export_dir}/{self.export_filename}_accuracy_metrics.csv', index=False)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot ECG signal with peaks
        plt.subplot(211)
        plt.plot(timestamps, ecg_data, 'b-', label='ECG Signal', alpha=0.7)
        
        # Plot true peaks
        plt.plot(true_peak_times, true_peak_values, 'go', label='True R-peaks', markersize=8)
        
        # Plot real-time detected peaks
        detected_times = self.export_data['true_peak_times']
        detected_values = [ecg_data[np.abs(timestamps - t).argmin()] for t in detected_times]
        plt.plot(detected_times, detected_values, 'rx', label='Real-time Detected', markersize=8)
        
        plt.title('ECG Signal with True and Detected R-peaks')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        # Plot timing error histogram
        plt.subplot(212)
        matched_delays = [d for d in self.export_data['detection_delays'] if d is not None]
        if matched_delays:
            timing_errors = np.array(matched_delays) * 1000  # Convert to ms
            plt.hist(timing_errors, bins=30, color='blue', alpha=0.7)
            plt.axvline(np.mean(timing_errors), color='r', linestyle='dashed',
                    label=f'Mean Error: {np.mean(timing_errors):.2f} ms')
            plt.title('Distribution of Detection Timing Errors')
            plt.xlabel('Detection Error (ms)')
            plt.ylabel('Frequency')
            plt.legend()
        
        # Add metrics text box
        metrics_text = (
            f"Detection Metrics:\n"
            f"Mean Error: {accuracy_metrics['Mean Detection Delay']:.2f} ms\n"
            f"Std Error: {accuracy_metrics['Std Detection Delay']:.2f} ms\n"
            f"Sensitivity: {accuracy_metrics['Sensitivity (Recall)']:.1f}%\n"
            f"Precision: {accuracy_metrics['Precision']:.1f}%\n"
            f"False Positives: {accuracy_metrics['False Positives']}\n"
            f"False Negatives: {accuracy_metrics['False Negatives']}"
        )
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{self.export_dir}/{self.export_filename}_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def run(self):
        """Main processing loop with enhanced timing analysis and data export"""
        try:
            print("Starting real-time ECG processing...")
            print("Press 'q' to quit and view analysis results.")
            
            while True:
                if keyboard.is_pressed('q'):
                    print("\nTerminating the processing...")
                    break

                sample, timestamp = self.inlet.pull_sample()
                is_r_peak, phase, delays = self.process_sample(sample[0], timestamp)
                
                if is_r_peak:
                    current_delay, avg_delay = delays
                    print(f"R-peak {self.total_detections} detected! | "
                          f"Current Delay: {current_delay*1000:.2f} ms | "
                          f"Average Delay: {avg_delay*1000:.2f} ms")
                
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Stopping the process...")
        
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
        
        finally:
            # Export results and generate analysis
            if hasattr(self, 'full_ecg_data') and len(self.full_ecg_data) > 0:
                print("\nGenerating analysis results...")
                self.export_results()
            else:
                print("\nNo data collected. Analysis cannot be performed.")
                
                
def main():
    """Main function to run the ECG processor"""
    try:
        processor = RealtimeECGProcessor()
        processor.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please ensure the LSL stream is running and try again.")

if __name__ == "__main__":
    main()