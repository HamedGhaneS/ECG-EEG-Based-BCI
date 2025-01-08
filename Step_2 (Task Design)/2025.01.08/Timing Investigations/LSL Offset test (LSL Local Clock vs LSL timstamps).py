from pylsl import StreamInlet, resolve_stream, local_clock
import time
import numpy as np
from datetime import datetime

class LSLTimingDiagnostic:
    def __init__(self):
        print("LSL Timing Diagnostic Tool")
        print("=" * 50)
        
        # Connect to R-peak stream
        print("\nLooking for R-peak stream...")
        streams = resolve_stream('type', 'R_PEAK')
        if not streams:
            raise RuntimeError("No R-peak stream found!")
        
        self.inlet = StreamInlet(streams[0])
        
        # Get stream info
        info = self.inlet.info()
        print(f"\nConnected to stream:")
        print(f"Name: {info.name()}")
        print(f"Type: {info.type()}")
        print(f"Source ID: {info.source_id()}")
        
        # Initialize timing statistics
        self.offsets = []
        self.time_corrections = []
        self.rr_intervals = []
        self.local_intervals = []
        self.timestamp_intervals = []

    def calculate_statistics(self, values):
        if not values:
            return None
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }

    def print_timing_stats(self):
        print("\nTiming Statistics:")
        print("=" * 50)
        
        # Offset statistics
        offset_stats = self.calculate_statistics(self.offsets)
        if offset_stats:
            print("\nOffset between local_clock() and LSL timestamps (ms):")
            print(f"Mean: {offset_stats['mean']*1000:.3f}")
            print(f"Std:  {offset_stats['std']*1000:.3f}")
            print(f"Min:  {offset_stats['min']*1000:.3f}")
            print(f"Max:  {offset_stats['max']*1000:.3f}")
        
        # Time correction statistics
        correction_stats = self.calculate_statistics(self.time_corrections)
        if correction_stats:
            print("\nLSL Time Correction Values (ms):")
            print(f"Mean: {correction_stats['mean']*1000:.3f}")
            print(f"Std:  {correction_stats['std']*1000:.3f}")
            print(f"Min:  {correction_stats['min']*1000:.3f}")
            print(f"Max:  {correction_stats['max']*1000:.3f}")
        
        # Interval comparisons
        if self.rr_intervals:
            print("\nInterval Comparisons (seconds):")
            print("Local Clock Intervals:")
            local_stats = self.calculate_statistics(self.local_intervals)
            print(f"Mean: {local_stats['mean']:.3f}")
            print(f"Std:  {local_stats['std']:.3f}")
            
            print("\nLSL Timestamp Intervals:")
            ts_stats = self.calculate_statistics(self.timestamp_intervals)
            print(f"Mean: {ts_stats['mean']:.3f}")
            print(f"Std:  {ts_stats['std']:.3f}")

    def run_diagnostic(self, duration=30):
        print(f"\nRunning diagnostic for {duration} seconds...")
        print("=" * 50)
        
        start_time = time.time()
        last_local = None
        last_timestamp = None
        sample_count = 0
        
        try:
            while (time.time() - start_time) < duration:
                # Get current local clock time
                current_local = local_clock()
                
                # Get time correction
                time_correction = self.inlet.time_correction()
                self.time_corrections.append(time_correction)
                
                # Get sample and its timestamp
                sample, timestamp = self.inlet.pull_sample()
                
                # Calculate intervals if we have previous measurements
                if last_local is not None:
                    self.local_intervals.append(current_local - last_local)
                    self.timestamp_intervals.append(timestamp - last_timestamp)
                
                # Store current times for next interval calculation
                last_local = current_local
                last_timestamp = timestamp
                
                # Calculate and store offset
                offset = timestamp - current_local
                self.offsets.append(offset)
                
                # Print current sample information
                sample_count += 1
                print(f"\nSample {sample_count}:")
                print(f"Local Clock:     {current_local:.6f}")
                print(f"LSL Timestamp:   {timestamp:.6f}")
                print(f"Offset (ms):     {offset*1000:.3f}")
                print(f"Time Correction: {time_correction*1000:.3f} ms")
                print("-" * 50)
                
        except KeyboardInterrupt:
            print("\nDiagnostic interrupted by user")
        
        print("\nDiagnostic complete")
        self.print_timing_stats()
        
        # Calculate clock drift
        if len(self.offsets) > 1:
            drift_rate = (self.offsets[-1] - self.offsets[0]) / (len(self.offsets) - 1)
            print(f"\nEstimated clock drift rate: {drift_rate*1000:.3f} ms/sample")

if __name__ == "__main__":
    diagnostic = LSLTimingDiagnostic()
    diagnostic.run_diagnostic(duration=30)
