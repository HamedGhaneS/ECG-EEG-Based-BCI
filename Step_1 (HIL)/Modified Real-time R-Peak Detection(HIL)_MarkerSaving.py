"""
Real-time ECG R-Peak Detection with LSL Integration and Timestamp Recording
------------------------------------------------------------------------
Author: Hamed Ghane
Date: November 22, 2024

This script performs real-time R-peak detection from ECG data streamed via LSL,
with precise timestamp recording of marker sending times. Compatible with both
BrainVision amplifier streams and simulated ECG data.

Features:
- Real-time R-peak detection with adaptive thresholding
- LSL marker generation for detected R-peaks
- Dual timestamp recording (Local PC time and LSL time)
- Signal quality monitoring
- Performance statistics and visualization
- Automatic CSV generation of marker timestamps

Timestamp Recording:
- Local PC Time (time.time())
- LSL Time (local_clock())
- Heart Rate at marker time
- Detection delay

Output:
- Real-time console feedback
- CSV file with marker timestamps
- Performance visualization
- Statistics summary

Usage:
1. Set USE_SIMULATION = True for simulated data, False for BrainVision
2. Run the script
3. Press Ctrl+C to stop and save data
"""

import numpy as np
from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet, local_clock
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, lfilter_zi, welch
from collections import deque
import csv

class SignalQuality:
    def __init__(self, window_size=1000):
        self.window = deque(maxlen=window_size)
        self.noise_threshold = None
        self.snr_threshold = 3.0
        
    def update(self, sample):
        self.window.append(sample)
        
    def compute_snr(self):
        if len(self.window) < self.window.maxlen:
            return float('inf')
        signal = np.array(self.window)
        noise = signal - np.convolve(signal, np.ones(10)/10, mode='same')
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        return 10 * np.log10(signal_power/noise_power) if noise_power > 0 else float('inf')

class AdaptiveFilter:
    def __init__(self, fs):
        self.fs = fs
        self.update_filter_parameters()
        
    def update_filter_parameters(self, low=5, high=15):
        nyq = self.fs / 2
        low_norm = low / nyq
        high_norm = high / nyq
        self.b, self.a = butter(1, [low_norm, high_norm], btype='band')
        self.zi = lfilter_zi(self.b, self.a) * 0

    def filter(self, x):
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y

    def optimize_filter(self, data):
        freqs, psd = welch(data, self.fs, nperseg=min(len(data), self.fs))
        qrs_mask = (freqs >= 2) & (freqs <= 40)
        if np.any(qrs_mask):
            peak_freq = freqs[qrs_mask][np.argmax(psd[qrs_mask])]
            self.update_filter_parameters(
                max(1, peak_freq-5),
                min(45, peak_freq+15)
            )

class ECGProcessor:
    def __init__(self, fs):
        self.fs = fs
        self.filter = AdaptiveFilter(fs)
        self.quality_monitor = SignalQuality()
        
        # Physiological constraints
        self.min_rr_interval = 0.25  # 240 BPM max
        self.max_rr_interval = 2.0   # 30 BPM min
        self.refractory_period = 0.2  # 200ms
        self.refractory_samples = int(self.refractory_period * fs)
        
        # Detection state
        self.last_peak_time = 0
        self.peak_heights = deque(maxlen=8)
        self.filtered_buffer = deque(maxlen=int(0.1 * fs))
        self.rr_intervals = deque(maxlen=8)
        self.baseline = 0
        self.noise_level = 0
        self.threshold = None
        self.last_valid_rr = None
        self.peaks_detected = 0
        
        # Marker control
        self.last_marker_peak_id = -1
        self.last_peak_processed = False
    def calibrate(self, data, duration=10.0):
        print("\nPerforming extended calibration...")
        
        self.baseline = np.median(data)
        centered_data = data - self.baseline
        self.filter.optimize_filter(centered_data)
        filtered_data = self.filter.filter(centered_data)
        
        noise = np.abs(np.diff(filtered_data))
        self.noise_level = np.median(noise) * 1.4826  # MAD estimate
        
        signal_std = np.std(filtered_data)
        peak_candidates = filtered_data[filtered_data > signal_std]
        if len(peak_candidates) > 0:
            self.threshold = np.percentile(peak_candidates, 25)
        else:
            self.threshold = signal_std * 0.8
        
        peaks = []
        last_peak = -self.refractory_samples
        
        for i in range(len(filtered_data)):
            if (i - last_peak) > self.refractory_samples:
                if filtered_data[i] > self.threshold:
                    window_start = max(0, i-10)
                    window_end = min(len(filtered_data), i+10)
                    if i == window_start + np.argmax(filtered_data[window_start:window_end]):
                        if self.validate_peak(filtered_data[i], i, filtered_data):
                            peaks.append(i)
                            last_peak = i
        
        print(f"Signal Statistics:")
        print(f"Baseline: {self.baseline:.2f}")
        print(f"Noise level: {self.noise_level:.2f}")
        print(f"Initial threshold: {self.threshold:.2f}")
        print(f"Found {len(peaks)} calibration peaks")
        
        if len(peaks) >= 2:
            rr_intervals = np.diff(peaks) / self.fs
            valid_rr = rr_intervals[(rr_intervals >= self.min_rr_interval) & 
                                  (rr_intervals <= self.max_rr_interval)]
            
            if len(valid_rr) > 0:
                self.last_valid_rr = np.median(valid_rr)
                print(f"Estimated HR: {60/self.last_valid_rr:.1f} BPM")
        
        return self.threshold
    
    def validate_peak(self, peak_value, peak_idx, data):
        if peak_value < self.noise_level * 3:
            return False
        
        window_start = max(0, peak_idx - int(0.1 * self.fs))
        window_end = min(len(data), peak_idx + int(0.1 * self.fs))
        window = data[window_start:window_end]
        local_noise = np.std(window)
        if local_noise > 0:
            snr = peak_value / local_noise
            if snr < 3:
                return False
        
        return True

    def process_sample(self, sample, timestamp):
        centered = sample - self.baseline
        filtered = self.filter.filter(np.array([centered]))[0]
        self.filtered_buffer.append(filtered)
        
        current_time = time.perf_counter()
        peak_detected = False
        detection_time = None
        marker_needed = False
        
        time_since_last_peak = current_time - self.last_peak_time
        
        if time_since_last_peak > self.refractory_period:
            if filtered > self.threshold:
                if len(self.filtered_buffer) >= 3:
                    if (self.filtered_buffer[-2] > self.filtered_buffer[-3] and 
                        self.filtered_buffer[-2] > self.filtered_buffer[-1]):
                        
                        if self.validate_peak(filtered, len(self.filtered_buffer)-2, 
                                           list(self.filtered_buffer)):
                            peak_detected = True
                            detection_time = current_time
                            self.last_peak_time = current_time
                            self.peaks_detected += 1
                            
                            if self.last_marker_peak_id < self.peaks_detected:
                                marker_needed = True
                                self.last_marker_peak_id = self.peaks_detected
                            
                            if self.last_valid_rr is not None:
                                if (self.min_rr_interval <= time_since_last_peak <= 
                                    self.max_rr_interval):
                                    self.rr_intervals.append(time_since_last_peak)
                                    self.last_valid_rr = time_since_last_peak
                            
                            self.peak_heights.append(filtered)
                            peak_mean = np.mean(list(self.peak_heights))
                            noise_est = self.noise_level
                            self.threshold = noise_est + 0.5 * (peak_mean - noise_est)
        
        return filtered, peak_detected, detection_time, marker_needed

def create_marker_outlet():
    info = StreamInfo(
        name='ECG_R_Peak_Markers',
        type='Markers',
        channel_count=1,
        nominal_srate=0,
        channel_format='string',
        source_id='rpeak_detector'
    )
    return StreamOutlet(info)

def save_timestamps(timestamps, filename=None):
    """Save the marker timestamps to a CSV file"""
    if filename is None:
        filename = f"marker_timestamps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Marker_Number', 'Local_Time', 'LSL_Time', 'HR', 'Detection_Delay_ms'])
        writer.writerows(timestamps)
    print(f"\nTimestamps saved to: {filename}")

def create_visualization(ecg_data, r_peaks, delays, fs):
    plt.figure(figsize=(15, 15))
    
    plt.subplot(311)
    time_array = np.arange(len(ecg_data)) / fs
    plt.plot(time_array, ecg_data, 'b-', label='ECG Signal', linewidth=0.5)
    if r_peaks:
        plt.plot(np.array(r_peaks) / fs, 
                [ecg_data[i] for i in r_peaks], 
                'ro', label='R-peaks')
    plt.title('ECG Signal with R-peaks')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(312)
    zoom_samples = min(int(5 * fs), len(ecg_data))
    plt.plot(time_array[:zoom_samples], ecg_data[:zoom_samples], 'b-', 
            label='ECG Signal', linewidth=0.5)
    zoom_peaks = [p for p in r_peaks if p < zoom_samples]
    if zoom_peaks:
        plt.plot(np.array(zoom_peaks) / fs,
                [ecg_data[i] for i in zoom_peaks],
                'ro', label='R-peaks')
    plt.title('First 5 Seconds')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(313)
    plt.plot(delays, 'g-', label='Detection Delay')
    plt.axhline(y=5, color='r', linestyle='--', label='5ms threshold')
    plt.title('Detection Delays')
    plt.xlabel('Peak Number')
    plt.ylabel('Delay (ms)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
def process_ecg_stream():
    print("Looking for ECG stream...")
    
    # Choose which stream to use (uncomment/comment as needed)
    USE_SIMULATION = True  # Set to False to use real BrainVision stream
    
    try:
        if USE_SIMULATION:
            print("Using simulated ECG stream...")
            streams = resolve_stream('name', 'ECGSimulation')
        else:
            print("Using real BrainVision stream...")
            streams = resolve_stream('name', 'BrainAmpSeries')
        
        inlet = StreamInlet(streams[0])
        
        # Get stream info
        stream_info = inlet.info()
        channel_count = stream_info.channel_count()
        fs = int(stream_info.nominal_srate())
        print(f"Connected to stream with {channel_count} channels at {fs} Hz")
        print("Using last channel for ECG data")
        
        processor = ECGProcessor(fs)
        marker_outlet = create_marker_outlet()
        print("Marker stream created: ECG_R_Peak_Markers")

        all_ecg_data = []
        all_r_peaks = []
        peak_detection_delays = []
        marker_timestamps = []  # Added: List to store timestamp information
        current_sample_index = 0
        processing_start_time = time.time()
        
        # Collect calibration data
        calibration_duration = 10
        print(f"\nCollecting {calibration_duration}s calibration data...")
        calibration_data = []
        for _ in range(int(fs * calibration_duration)):
            sample, timestamp = inlet.pull_sample()
            calibration_data.append(sample[-1])  # Always use last channel
        
        processor.calibrate(np.array(calibration_data))
        
        print("\nStarting real-time detection...")
        last_status_time = time.time()
        last_status_update = 0
        
        try:
            while True:
                sample, timestamp = inlet.pull_sample()
                sample_time = time.perf_counter()
                current_sample = sample[-1]  # Always use last channel
                
                filtered, peak_detected, detection_time, marker_needed = processor.process_sample(
                    current_sample, timestamp)
                
                all_ecg_data.append(current_sample)
                
                if peak_detected:
                    all_r_peaks.append(current_sample_index)
                    delay = (detection_time - sample_time) * 1000
                    peak_detection_delays.append(delay)
                    
                    if marker_needed:
                        local_time = time.time()
                        lsl_time = local_clock()
                        marker_outlet.push_sample(['R'])
                        hr = 60.0 / processor.last_valid_rr if len(processor.rr_intervals) > 0 else 0
                        marker_timestamps.append([processor.peaks_detected, local_time, lsl_time, hr, delay])
                        if len(processor.rr_intervals) > 0:
                            print(f"R-peak {processor.peaks_detected}: "
                                  f"HR={hr:.1f} BPM, Delay={delay:.2f}ms, Marker sent")
                
                current_time = time.time()
                if current_time - last_status_time >= 1.0:
                    elapsed_time = current_time - processing_start_time
                    processing_speed = current_sample_index/elapsed_time
                    
                    if abs(processing_speed - last_status_update) > 50:
                        print(f"Processing: {processing_speed:.1f} Hz")
                        last_status_update = processing_speed
                    last_status_time = current_time
                
                current_sample_index += 1
                
        except KeyboardInterrupt:
            print("\nProcessing complete...")
            
            total_time = time.time() - processing_start_time
            print(f"\nPerformance Statistics:")
            print(f"Processing Speed: {current_sample_index/total_time:.1f} Hz")
            print(f"Total Samples: {current_sample_index}")
            print(f"Total R-peaks: {processor.peaks_detected}")
            
            if peak_detection_delays:
                print(f"Average Delay: {np.mean(peak_detection_delays):.2f}ms")
                print(f"Max Delay: {np.max(peak_detection_delays):.2f}ms")
                print(f"Min Delay: {np.min(peak_detection_delays):.2f}ms")
            
            if len(processor.rr_intervals) >= 2:
                avg_hr = 60.0 / np.mean(list(processor.rr_intervals))
                print(f"Average Heart Rate: {avg_hr:.1f} BPM")
            
            # Added: Save timestamps before visualization
            if marker_timestamps:
                save_timestamps(marker_timestamps)
            
            create_visualization(all_ecg_data, all_r_peaks, peak_detection_delays, fs)
            
    except Exception as e:
        print(f"Error connecting to stream: {str(e)}")
        return

if __name__ == "__main__":
    try:
        process_ecg_stream()
    except KeyboardInterrupt:
        print("\nScript terminated by user")
    except Exception as e:
        print(f"\nUnexpected error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        print("\nScript ended")