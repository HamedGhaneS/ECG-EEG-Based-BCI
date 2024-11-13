"""
Script Title: Real-time ECG R-Peak Detection with Polarity Check and Visualization
Author: Hamed Ghane
Date: 2024-11-11

Description:
    This script implements real-time ECG signal processing with automatic polarity detection
    and R-peak identification. It processes incoming ECG data streams, detects signal polarity
    issues, and provides user control over signal processing decisions.

Features:
    - Real-time ECG signal processing with adjustable sampling rate
    - Automatic polarity detection with user-controlled response options
    - R-peak detection using modified Pan-Tompkins algorithm
    - Cardiac phase (systole/diastole) tracking
    - Real-time performance metrics and delay analysis
    - Data logging and visualization capabilities

Workflow:
    1. Initialization Phase:
       - Set up LSL stream connection for ECG data
       - Initialize signal processing parameters and buffers
       - Configure bandpass filter (0.5-40 Hz)

    2. Training Phase (First 10 seconds):
       - Collect initial ECG data for parameter calibration
       - Perform signal polarity check with user interaction
       - Calculate adaptive thresholds for R-peak detection
       - Establish baseline cardiac timing parameters

    3. Real-time Processing:
       - Continuous ECG data acquisition and filtering
       - R-peak detection with adaptive thresholding
       - Cardiac phase classification
       - Performance monitoring (detection delays)

    4. Data Management:
       - Continuous logging of ECG data and detection events
       - Real-time performance metrics calculation
       - Export of processed data and statistics

    5. Visualization:
       - Plot raw and processed ECG signals
       - Mark detected R-peaks
       - Display performance statistics

Dependencies:
    - NumPy: Numerical computations and array operations
    - PyLSL: Lab Streaming Layer interface
    - SciPy: Signal processing and filtering
    - Matplotlib: Data visualization
    - Pandas: Data structuring and export

Usage:
    The script processes real-time ECG data streams, automatically detecting signal
    polarity issues and providing users with options to:
    1. Stop for investigation
    2. Accept potential risks
    3. Automatically invert the signal

Output:
    - Real-time R-peak detection metrics
    - ECG signal plots with marked R-peaks
    - Detection performance statistics
    - Exported CSV files with processed data
"""

import numpy as np
from pylsl import StreamInlet, resolve_stream, local_clock
from scipy.signal import butter, sosfilt, sosfiltfilt
import time
from collections import deque
from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
import keyboard

class RealtimeECGProcessor:
    def __init__(self, sampling_rate=1000, training_duration=10):
        """Initialize the real-time ECG processor"""
        self.sampling_rate = sampling_rate
        self.training_duration = training_duration
        self.invert_signal = False  # Flag for signal inversion
        
        # Add the time domain difference check right here
        print(f"Time domain difference: {local_clock() - time.time()}")   
             
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
        
    def setup_data_export(self):
        """Setup data export structures and files"""
        self.export_data = {
            'true_peak_times': [],
            'detection_times': [],
            'detection_delays': [],
            'rr_intervals': [],
            'cardiac_phases': []
        }
        
        self.full_ecg_data = []
        
        self.export_dir = 'ecg_analysis_results'
        os.makedirs(self.export_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.export_filename = f'ecg_analysis_{timestamp}'
        
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

    def check_signal_polarity(self, signal_data):
        """Check signal polarity and prompt user if negative polarity is detected"""
        print("\nChecking signal polarity...")
        max_peak = np.max(signal_data)
        min_peak = np.min(signal_data)
        
        if abs(min_peak) > abs(max_peak):
            print("Warning: Detected potential negative polarity.")
            user_input = input("Options:\n1: Stop for investigation\n2: Accept risk\n3: Invert signal\nEnter choice: ")
            
            if user_input == '1':
                print("Stopping for investigation.")
                return False
            elif user_input == '2':
                print("Proceeding with detected polarity (negative).")
                return True
            elif user_input == '3':
                print("Inverting signal for analysis.")
                self.invert_signal = True
                return True
            else:
                print("Invalid choice. Stopping analysis.")
                return False
        else:
            print("Polarity check passed with positive peaks.")
            return True
    
    def train_detector(self):
        """Train the R-peak detector using initial 10-second data"""
        print("Training R-peak detector... Initial analysis and information extraction for threshold tuning.")
        
        # Get training data
        signal = np.array([x[0] for x in self.training_buffer])
        times = np.array([x[1] for x in self.training_buffer])

        # First perform polarity check
        if not self.check_signal_polarity(signal):
            raise RuntimeError("Signal polarity check failed. Please investigate the signal.")

        # Apply signal inversion if needed
        if self.invert_signal:
            signal = -signal
        
        # Filter signal
        filtered_signal = sosfiltfilt(self.sos, signal)
        
        # Process for R-peak detection
        diff_signal = np.diff(filtered_signal)
        squared_signal = diff_signal ** 2
        
        # Moving average
        window_size = int(0.150 * self.sampling_rate)
        ma_signal = np.convolve(squared_signal, np.ones(window_size)/window_size, mode='same')
        
        # Set thresholds
        self.qrs_peak_threshold = np.mean(ma_signal) + 2 * np.std(ma_signal)
        self.noise_threshold = np.mean(ma_signal) + np.std(ma_signal)
        
        # Initial R-peak detection
        peaks = self.find_r_peaks(ma_signal)
        
        # Calculate average RR interval
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
        """Process a new ECG sample in real-time and save it to the full log"""
        if self.is_training:
            self.training_buffer.append((sample, timestamp))
            if len(self.training_buffer) >= self.training_duration * self.sampling_rate:
                self.train_detector()
            return False, "unknown", None
        
        # Apply inversion if needed
        sample = -sample if self.invert_signal else sample
        
        # Log the sample for full ECG data
        self.full_ecg_data.append((timestamp, sample))
        
        self.signal_buffer.append((sample, timestamp))
        
        if len(self.signal_buffer) < self.r_peak_window * 2:
            return False, self.current_phase, None
        
        # Process latest window
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
                    
                    print(f"""
                    Timestamps (both in LSL time domain):
                    Sample arrival: {timestamp:.6f}
                    Detection:     {detection_time:.6f}
                    Delay:        {(detection_time - timestamp)*1000:.3f} ms
                    """)
                    
                    # Store timestamps
                    self.r_peak_times.append(timestamp)
                    self.detection_times.append(detection_time)
                    
                    # Calculate delays
                    current_delay = detection_time - timestamp
                    self.detection_delays.append(current_delay)
                    self.total_detections += 1
                    self.cumulative_delay += current_delay
                    
                    delays = (current_delay, self.cumulative_delay / self.total_detections)
                    
                    # Store data for export
                    self.export_data['true_peak_times'].append(timestamp)
                    self.export_data['detection_times'].append(detection_time)
                    self.export_data['detection_delays'].append(current_delay)
                    self.export_data['cardiac_phases'].append(self.current_phase)
                    
                    if len(self.export_data['true_peak_times']) > 1:
                        self.export_data['rr_intervals'].append(
                            timestamp - self.export_data['true_peak_times'][-2])
                    
                    # Update threshold and phase
                    self.qrs_peak_threshold = 0.875 * self.qrs_peak_threshold + 0.125 * squared_signal[-1]
                    self.current_phase = "systole"
        else:
            # Check if we should transition to diastole
            if self.r_peak_times and timestamp - self.r_peak_times[-1] > self.systole_duration:
                self.current_phase = "diastole"
        
        return is_r_peak, self.current_phase, delays

    def export_results(self):
        """Export analysis results to files and plot ECG with detected R-peaks"""
        # Save full ECG data
        ecg_df = pd.DataFrame(self.full_ecg_data, columns=['Timestamp', 'ECG_Signal'])
        ecg_df.to_csv(f'{self.export_dir}/{self.export_filename}_full_stream.csv', index=False)
        
        # Save R-peak detection results
        detection_df = pd.DataFrame({
            'True_Peak_Time': self.export_data['true_peak_times'],
            'Detection_Time': self.export_data['detection_times'],
            'Detection_Delay_ms': [d * 1000 for d in self.export_data['detection_delays']],
            'Cardiac_Phase': self.export_data['cardiac_phases']
        })
        detection_df.to_csv(f'{self.export_dir}/{self.export_filename}_detections.csv', index=False)
        
        # Calculate statistics
        avg_delay = np.mean(self.export_data['detection_delays']) * 1000 if self.total_detections > 0 else 0
        avg_rr_interval = np.mean(self.export_data['rr_intervals']) * 1000 if len(self.export_data['rr_intervals']) > 0 else 0
        
        # Print statistics
        print("\n--- Final Statistics ---")
        print(f"Total R-Peaks Detected: {self.total_detections}")
        print(f"Average Detection Delay: {avg_delay:.2f} ms")
        print(f"Average R-R Interval Duration: {avg_rr_interval:.2f} ms")
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(ecg_df['Timestamp'], ecg_df['ECG_Signal'], label='ECG Signal')
        
        # Mark detected R-peaks
        r_peak_times = self.export_data['true_peak_times']
        r_peak_values = [ecg_df[ecg_df['Timestamp'] == time]['ECG_Signal'].values[0] for time in r_peak_times]
        plt.scatter(r_peak_times, r_peak_values, color='red', marker='x', label='R-peaks')
        
        plt.title(f"ECG Signal with Detected R-Peaks\n"
                f"Avg Detection Delay: {avg_delay:.2f} ms | "
                f"Total R-Peaks Detected: {self.total_detections} | "
                f"Avg R-R Interval: {avg_rr_interval:.2f} ms")
        plt.xlabel('Time (s)')
        plt.ylabel('ECG Signal Amplitude')
        plt.legend()
        plt.show()
        
        print(f"\nResults exported to {self.export_dir}/{self.export_filename}_*.csv")

    def run(self):
        """Main processing loop with enhanced timing analysis and data export"""
        try:
            print("Starting real-time ECG processing... Press 'q' to quit.")
            while True:
                if keyboard.is_pressed('q'):
                    print("\n'Q' pressed. Terminating the script...")
                    break

                sample, timestamp = self.inlet.pull_sample()
                is_r_peak, phase, delays = self.process_sample(sample[0], timestamp)
                
                if is_r_peak:
                    current_delay, cumulative_delay = delays
                    print(f"R-peak {self.total_detections} detected! | "
                          f"Current Delay: {current_delay*1000:.2f} ms | "
                          f"Average Delay: {cumulative_delay*1000:.2f} ms")
                
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Stopping the process.")
        
        finally:
            self.export_results()

def main():
    processor = RealtimeECGProcessor()
    processor.run()

if __name__ == "__main__":
    main()
