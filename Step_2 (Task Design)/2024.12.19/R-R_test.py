from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
import csv
from datetime import datetime

class HeartbeatMonitor:
    def __init__(self):
        # Initialize settings for R-R interval validation
        self.rr_min = 0.6  # Minimum physiological R-R interval in seconds
        self.rr_max = 1.2  # Maximum physiological R-R interval in seconds
        self.r_peaks = []  # Store R-peak timestamps
        self.intervals = []  # Store R-R intervals
        
        # Create CSV file with timestamp in name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.filename = f"rpeak_data_{timestamp}.csv"
        
        # Initialize CSV with headers
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Interval', 'Heart_Rate'])

    def connect_to_stream(self):
        """Connect to the LSL stream for R-peak markers"""
        print("Looking for R-peak stream...")
        streams = resolve_stream('type', 'R_PEAK')
        if not streams:
            raise RuntimeError("No R-peak stream found!")
        
        self.inlet = StreamInlet(streams[0])
        print("Connected to R-peak stream!")

    def calculate_interval(self, new_peak):
        """Calculate interval between consecutive R-peaks"""
        if self.r_peaks:
            interval = new_peak - self.r_peaks[-1]
            if self.rr_min <= interval <= self.rr_max:
                heart_rate = 60.0 / interval  # Calculate heart rate in BPM
                return interval, heart_rate
        return None, None

    def run(self, duration=60):
        """Run the monitor for specified duration in seconds"""
        print(f"Starting R-peak monitoring for {duration} seconds...")
        start_time = time.time()
        
        try:
            while (time.time() - start_time) < duration:
                # Get sample from stream
                sample, timestamp = self.inlet.pull_sample(timeout=0.0)
                
                if sample is not None:
                    # Calculate interval
                    interval, heart_rate = self.calculate_interval(timestamp)
                    
                    # Store data
                    self.r_peaks.append(timestamp)
                    
                    # Print and save data
                    if interval is not None:
                        self.intervals.append(interval)
                        print(f"R-peak detected - Interval: {interval:.3f}s, Heart Rate: {heart_rate:.1f} BPM")
                        
                        # Save to CSV
                        with open(self.filename, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([timestamp, interval, heart_rate])
                    else:
                        print(f"R-peak detected at {timestamp:.3f}")
                
                time.sleep(0.001)  # Prevent CPU overload
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        
        # Print summary statistics
        if self.intervals:
            print("\nSummary Statistics:")
            print(f"Number of R-peaks detected: {len(self.r_peaks)}")
            print(f"Average R-R interval: {np.mean(self.intervals):.3f}s")
            print(f"Average Heart Rate: {60.0/np.mean(self.intervals):.1f} BPM")
            print(f"Data saved to: {self.filename}")

if __name__ == "__main__":
    # Create and run monitor
    monitor = HeartbeatMonitor()
    monitor.connect_to_stream()
    monitor.run(duration=60)  # Monitor for 60 seconds
