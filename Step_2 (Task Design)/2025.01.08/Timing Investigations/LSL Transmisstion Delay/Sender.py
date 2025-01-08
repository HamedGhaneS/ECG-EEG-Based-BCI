from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream, local_clock
import time
import numpy as np

class LSLLatencyTester:
    def __init__(self):
        # First, we create our outlet for sending test markers
        # This needs to happen before we look for the receiver's stream
        print("Creating marker outlet...")
        self.marker_info = StreamInfo(
            'LatencyTest',      # Name of the stream
            'Markers',          # Type of the stream
            1,                  # Number of channels
            0,                  # Irregular sampling rate (events)
            'string',           # Data format
            'latency_test_out'  # Unique source ID
        )
        self.marker_outlet = StreamOutlet(self.marker_info)
        print("Marker outlet created")

        # Allow time for network discovery
        # This delay helps ensure our stream is visible on the network
        print("Waiting 5 seconds for streams to be discovered...")
        time.sleep(5)

        # Now we can look for the receiver's acknowledgment stream
        # We'll keep trying until we find it
        print("Looking for acknowledgment stream...")
        while True:
            streams = resolve_stream('name', 'LatencyAck')
            if streams:
                self.ack_inlet = StreamInlet(streams[0])
                print("Connected to acknowledgment stream")
                break
            print("No acknowledgment stream found, retrying in 1 second...")
            time.sleep(1)

        # Initialize storage for our timing measurements
        self.latencies = []

    def run_test(self, n_trials=1000):
        print(f"\nRunning latency test with {n_trials} trials...")
        print("=" * 50)

        try:
            for i in range(n_trials):
                # Record the exact time before sending marker
                send_time = local_clock()

                # Send our test marker
                self.marker_outlet.push_sample(['ping'])

                # Wait for acknowledgment with a 1-second timeout
                # This prevents hanging if network issues occur
                sample, timestamp = self.ack_inlet.pull_sample(timeout=1.0)
                if sample is None:
                    print(f"Warning: Trial {i+1} timed out waiting for acknowledgment")
                    continue

                # Record the time when we receive the acknowledgment
                receive_time = local_clock()

                # Calculate timing
                # Round-trip time is total time for marker to go to receiver and back
                round_trip = receive_time - send_time
                # Estimate one-way time by dividing round-trip by 2
                one_way = round_trip / 2
                # Store the result in milliseconds for easier reading
                self.latencies.append(one_way * 1000)

                # Show progress periodically
                if (i + 1) % 100 == 0:
                    print(f"Completed {i + 1} trials")

                # Small delay between trials to prevent flooding the network
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        finally:
            # Always show results, even if test was interrupted
            self.print_results()

    def print_results(self):
        if not self.latencies:
            print("No latency measurements collected")
            return

        latencies = np.array(self.latencies)

        print("\nLatency Test Results")
        print("=" * 50)
        print(f"Number of trials: {len(self.latencies)}")
        print(f"Mean latency: {np.mean(latencies):.3f} ms")
        print(f"Median latency: {np.median(latencies):.3f} ms")
        print(f"Std deviation: {np.std(latencies):.3f} ms")
        print(f"Min latency: {np.min(latencies):.3f} ms")
        print(f"Max latency: {np.max(latencies):.3f} ms")

        # Calculate percentiles to understand the distribution of latencies
        percentiles = [1, 5, 25, 75, 95, 99]
        print("\nPercentiles:")
        for p in percentiles:
            value = np.percentile(latencies, p)
            print(f"{p}th percentile: {value:.3f} ms")

if __name__ == "__main__":
    tester = LSLLatencyTester()
    tester.run_test()
