from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
import time

class LSLLatencyReceiver:
    def __init__(self):
        # Create outlet for acknowledgments first
        print("Creating acknowledgment outlet...")
        self.ack_info = StreamInfo(
            'LatencyAck',
            'Markers',
            1,
            0,
            'string',
            'latency_test_ack'
        )
        self.ack_outlet = StreamOutlet(self.ack_info)
        print("Acknowledgment outlet created")

        # Give time for streams to be discovered
        print("Waiting 5 seconds for streams to be discovered...")
        time.sleep(5)

        # Now wait for test marker stream
        print("Looking for test marker stream...")
        while True:
            streams = resolve_stream('name', 'LatencyTest')
            if streams:
                self.marker_inlet = StreamInlet(streams[0])
                print("Connected to test marker stream")
                break
            print("No test marker stream found, retrying in 1 second...")
            time.sleep(1)

    def run(self):
        print("\nStarting latency test receiver...")
        print("Waiting for markers and sending acknowledgments...")
        print("Press Ctrl+C to stop")

        try:
            while True:
                # Wait for marker with timeout
                sample, timestamp = self.marker_inlet.pull_sample(timeout=1.0)
                if sample is not None:
                    # Immediately send acknowledgment
                    self.ack_outlet.push_sample(['ack'])

        except KeyboardInterrupt:
            print("\nReceiver stopped by user")

if __name__ == "__main__":
    receiver = LSLLatencyReceiver()
    receiver.run()
