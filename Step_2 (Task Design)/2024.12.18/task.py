import random 
from psychopy import visual, core, data, event, gui
import pandas as pd
import numpy as np
from pathlib import Path
from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet
import threading
import queue
import time
try:
    import pyxid2 as pyxid
except ImportError:
    import pyxid

class CardiacSyncedLearningTask:
    def __init__(self):
        # Task settings
        self.settings = {
            'decision_duration': 1.25,  # Keep existing timing
            'delay_min': 1.5,
            'delay_max': 2.0,
            'feedback_duration': 4.0,
            'iti_min': 1.0,
            'iti_max': 1.5,
            'win_probability_good': 0.70,
            'win_probability_bad': 0.30,
            # Remove the cardiac window settings and add new timing structure
            'timing_conditions': {
                'systole': {
                    1: 10,   # Block 1: Early systole
                    2: 150,  # Block 2: Mid systole
                    3: 290   # Block 3: Late systole
                },
                'diastole': {
                    4: 310,  # Block 4: Early diastole
                    5: 500,  # Block 5: Mid diastole
                    6: 690   # Block 6: Late diastole
                }
            }
        }
        
        # Add this line right after settings initialization
        self.block_timing_map = self.create_randomized_block_order()
        
        # Setup Cedrus box first to ensure it's ready before other initializations
        self.cedrus_box = None
        try:
            devices = pyxid.get_xid_devices()
            if devices:
                self.cedrus_box = devices[0]
                self.cedrus_box.reset_rt_timer()
                self.cedrus_box.clear_response_queue()
                print("Cedrus device initialized successfully")
            else:
                print("No Cedrus devices found")
        except Exception as e:
            print(f"Error initializing Cedrus device: {e}")

        # Setup LSL outlet for markers
        self.marker_info = StreamInfo(
            'TaskMarkers',     # Stream name
            'Markers',         # Stream type
            1,                 # Number of channels
            0,                 # Irregular sampling rate
            'string',          # Channel format
            'TaskMarker123'    # Source id
        )
        self.marker_outlet = StreamOutlet(self.marker_info)

        # Marker codes [previous marker codes remain the same]
        self.markers = {
            'experiment_start': 'exp_start',
            'experiment_end': 'exp_end',
            'block_start': 'block_start',
            'block_end': 'block_end',
            'reversal': 'reversal',
            'trial_start': 'trial_start',
            'choice_made': 'choice_made',
            'feedback_onset': 'feedback_onset',
            'trial_end': 'trial_end',
            'instruction_start': 'instruct_start',
            'instruction_end': 'instruct_end',
            'timeout': 'timeout',
            'win_feedback': 'feedback_win',
            'loss_feedback': 'feedback_loss',
            'neutral_feedback': 'feedback_neutral',
            'break_start': 'break_start',
            'break_end': 'break_end'

        }

        # Setup paths
        self.base_path = Path.cwd()
        self.stim_path = self.base_path / 'stimuli'
        self.data_path = self.base_path / 'data'
        self.data_path.mkdir(exist_ok=True)

        # Define stimuli filenames
        self.stimuli = {
            'symbols': ['A.png', 'B.png', 'C.png'],
            'fixation': 'fixation.png',
            'feedback': {
                'win': 'win.png',
                'loss': 'loss.png',
                'neutral': 'NEU.png'
            }
        }

        # Task instructions
        self.instructions = [
            """Welcome to the experiment!

    In this task, you will see two symbols on each trial.
    Your goal is to learn which symbol is more likely to give you rewards.
    Note that these probabilities may change during the task.

    Press YELLOW BUTTON to continue...""",

            """On each trial:
    1. Two symbols will appear on the screen
    2. Choose the left symbol with the LEFT ARROW key
    3. Choose the right symbol with the RIGHT ARROW key
    4. You have 1.25 seconds to make your choice

    Press YELLOW BUTTON to continue...""",

            """After your choice:
    - If you see an UPWARD ARROW, your choice was Rewarded
    - If you see a DOWNWARD ARROW, your choice was Not Rewarded
    - The same symbols will have different probabilities of being Rewarded
    - These probabilities will change during each block

    Press YELLOW BUTTON to continue...""",

            """Important Notes:
    - Please respond before the symbols disapear from the screen
    - If you don't respond in time, you'll see a warning message
    - You can press 'BLUE KEY' at any time to end the experiment
    - Your data will be saved automatically

    Press YELLOW BUTTON to begin the task."""
        ]

        # Initialize queue for R-peak times
        self.r_peak_times = queue.Queue()

    def create_randomized_block_order(self):
        """Create a mapping of block numbers to randomized timing conditions"""
        import numpy as np

        # Create list of timing conditions
        timing_conditions = [
            ('systole', 10),    # Condition 1: Early systole
            ('systole', 150),   # Condition 2: Mid systole
            ('systole', 290),   # Condition 3: Late systole
            ('diastole', 310),  # Condition 4: Early diastole
            ('diastole', 500),  # Condition 5: Mid diastole
            ('diastole', 690)   # Condition 6: Late diastole
        ]

        # Shuffle the timing conditions
        np.random.shuffle(timing_conditions)

        # Create mapping of block numbers (1-6) to shuffled conditions
        timing_map = {}
        for block_num, (phase, timing) in enumerate(timing_conditions, 1):
            timing_map[block_num] = {
                'original_block': block_num,  # Keep block numbers in order
                'phase': phase,
                'timing': timing
            }

        return timing_map
    
    
    def cleanup_cedrus(self):
        """Safely cleanup Cedrus device connection"""
        try:
            if hasattr(self, 'cedrus_box') and self.cedrus_box:
                self.cedrus_box.clear_response_queue()
                time.sleep(0.5)  # Give device time to clear
        except Exception as e:
            print(f"Error during Cedrus cleanup: {e}")

    def generate_fixed_outcomes(self, n_trials, win_probability):
        """
        Generate a fixed sequence of outcomes that exactly matches the desired probability
        for a specific number of trials.

        Parameters:
        - n_trials: Exact number of trials needed
        - win_probability: Desired probability of wins

        Returns:
        List of boolean values with exact probability match
        """
        n_wins = round(n_trials * win_probability)
        outcomes = ([True] * n_wins) + ([False] * (n_trials - n_wins))
        np.random.shuffle(outcomes)
        return outcomes

    def initialize_block_outcomes(self, n_trials, reversal_point):
        """
        Initialize predetermined outcomes for a block with exact probability matching
        for both pre- and post-reversal segments.

        Parameters:
        - n_trials: Total number of trials in the block
        - reversal_point: The trial number where probability reversal occurs

        Returns:
        Dictionary containing separate outcome arrays for pre- and post-reversal periods,
        each maintaining exact probability ratios.
        """
        # Calculate the number of trials in each segment
        pre_reversal_trials = reversal_point
        post_reversal_trials = n_trials - reversal_point

        # Generate outcomes for first half (pre-reversal) with exact probabilities
        symbol_a_first = self.generate_fixed_outcomes(pre_reversal_trials, 
                                                    self.settings['win_probability_good'])
        symbol_b_first = self.generate_fixed_outcomes(pre_reversal_trials, 
                                                    self.settings['win_probability_bad'])

        # Generate outcomes for second half (post-reversal) with exact probabilities
        symbol_a_second = self.generate_fixed_outcomes(post_reversal_trials, 
                                                     self.settings['win_probability_bad'])
        symbol_b_second = self.generate_fixed_outcomes(post_reversal_trials, 
                                                     self.settings['win_probability_good'])

        return {
            'first_half': {
                0: symbol_a_first,  # Symbol A pre-reversal outcomes
                1: symbol_b_first   # Symbol B pre-reversal outcomes
            },
            'second_half': {
                0: symbol_a_second,  # Symbol A post-reversal outcomes
                1: symbol_b_second   # Symbol B post-reversal outcomes
            }
        }

    def send_marker(self, marker_code):
        """Send an LSL marker with the specified code"""
        self.marker_outlet.push_sample([marker_code])
        print(f"LSL Marker sent: {marker_code}")

    def setup_experiment(self):
        """Initialize PsychoPy window and load stimuli"""
        self.win = visual.Window(
            size=[1000, 800],
            fullscr=False,
            units='height',
            color=[0, 0, 0],
            allowGUI=True
        )

        # Load stimuli
        self.stim = {
            'symbols': [
                visual.ImageStim(self.win, image=str(self.stim_path / img))
                for img in self.stimuli['symbols'][:2]  # Only load A and B
            ],
            'fixation': visual.ImageStim(
                self.win,
                image=str(self.stim_path / self.stimuli['fixation']),
                pos=(0, 0)
            ),
            'feedback': {
                key: visual.ImageStim(self.win, image=str(self.stim_path / path))
                for key, path in self.stimuli['feedback'].items()
            },
            'text': visual.TextStim(
                self.win,
                text='',
                height=0.05,
                wrapWidth=0.8,
                color='white'
            ),
            
            'break_msg': visual.TextStim(
                self.win,
                text='You did well! You can rest for 2 minutes. Press the YELLOW BUTTON to start the next block.',
                height=0.05,
                wrapWidth=0.8,
                color='white'
            ),

            'timeout_msg': visual.TextStim(
                self.win,
                text='Too Slow',
                height=0.05,
                wrapWidth=0.8,
                color='black'
            )
        }

    def show_break_message(self):
        """Show a break message between blocks and wait for 2 minutes or until the yellow button is pressed."""
        self.cedrus_box.clear_response_queue()  # Clear any pending responses
        self.stim['break_msg'].draw()
        self.win.flip()
        self.send_marker('break_start')

        break_timer = core.Clock()
        break_timer.reset()
        while break_timer.getTime() < 120:  # 2 minutes break
            # Check Cedrus response
            self.cedrus_box.poll_for_response()
            if self.cedrus_box.response_queue:
                response = self.cedrus_box.get_next_response()
                if response['pressed'] and response['key'] == 5:  # Yellow button to continue
                    break
            core.wait(0.001)  # Prevent CPU overload

        self.cedrus_box.clear_response_queue()  # Clear responses before leaving break
        self.send_marker('break_end')

    
    def setup_lsl(self):
        """Setup LSL inlet for R-peak markers with proper time correction"""
        print("Looking for R-peak markers stream...")
        try:
            streams = resolve_stream('type', 'R_PEAK')
            if not streams:
                raise RuntimeError("No R-peak stream found")
            self.inlet = StreamInlet(streams[0])

            # Clear any existing queue
            self.r_peak_times = queue.Queue()

            # Store initial time correction
            self.time_correction = self.inlet.time_correction()
            print("R-peak stream found and connected!")

            # Start R-peak collection thread
            self.lsl_thread = threading.Thread(target=self.collect_r_peaks)
            self.lsl_thread.daemon = True
            self.lsl_thread.start()
            print("R-peak collection thread started")

            # Wait for first R-peak to confirm stream is working
            try:
                _ = self.r_peak_times.get(timeout=5)
                print("R-peak detection confirmed")
            except queue.Empty:
                raise RuntimeError("No R-peaks detected in first 5 seconds")

        except Exception as e:
            raise RuntimeError(f"Failed to setup LSL stream: {str(e)}")  

    def collect_r_peaks(self):
        """Continuously collect R-peak markers with improved handling"""
        while True:
            try:
                # Add timeout to prevent tight loop
                sample, timestamp = self.inlet.pull_sample(timeout=0.1)
                if sample is not None:
                    # Only add to queue if we're not overflowing
                    if self.r_peak_times.qsize() < 10:  # Keep queue manageable
                        self.r_peak_times.put(timestamp)
                    else:
                        # Clear queue if it's getting too full
                        while not self.r_peak_times.empty():
                            _ = self.r_peak_times.get()
                else:
                    # No sample available, give system time to breathe
                    time.sleep(0.001)
            except Exception as e:
                print(f"R-peak collection warning: {e}")
                time.sleep(0.001)  # Prevent rapid error loops

    def wait_for_timing(self, target_time, window=0.005):
        """More efficient timing wait with system consideration

        Args:
            target_time (float): Target time in LSL timebase
            window (float): Fine-tuning window in seconds
        """
        current_time = self.get_lsl_time()

        # If we're already past the target, return immediately
        if current_time >= target_time:
            return

        # First phase: coarse waiting with system breaks
        while (target_time - self.get_lsl_time()) > window:
            core.wait(0.001)  # Give system time to breathe

        # Second phase: fine-tuning
        while self.get_lsl_time() < target_time:
            pass  # Tight loop only for final milliseconds
    
    def get_participant_info(self):
        """Show dialog to collect participant information"""
        current_time = time.strftime("%Y%m%d-%H%M%S")

        exp_info = {
            'participant': '',
            'session': '001',
            'run': '1',
            'n_blocks': 6,      # Fixed at 6 blocks
            'n_trials': 10,     # Fixed at 40 trials - 
            'date_time': current_time,
        }

        dlg = gui.DlgFromDict(
            dictionary=exp_info,
            title='Task Info',
            fixed=['date_time', 'n_blocks', 'n_trials']  # Make blocks and trials fixed
        )

        if dlg.OK:
            return exp_info
        else:
            core.quit()

    def show_instructions(self):
        """Display task instructions with markers"""
        self.send_marker(self.markers['instruction_start'])

        for instruction in self.instructions:
            self.stim['text'].text = instruction
            self.stim['text'].draw()
            self.win.flip()

            while True:
                # Check Cedrus response
                self.cedrus_box.poll_for_response()
                if self.cedrus_box.response_queue:
                    response = self.cedrus_box.get_next_response()
                    if response['pressed']:  # Only handle button press, not release
                        if response['key'] == 5:  # Yellow button to continue
                            break
                        elif response['key'] == 6:  # Blue button to quit
                            self.cleanup_cedrus()
                            self.win.close()
                            core.quit()

                core.wait(0.001)  # Prevent CPU overload

        self.send_marker(self.markers['instruction_end'])

    def get_block_timing(self, current_block):
        """Get the timing parameters for the current block"""
        block_info = self.block_timing_map[current_block + 1]  # +1 because blocks are 0-indexed in the loop
        return block_info['timing'], block_info['phase']

    def get_lsl_time(self):
        """Get current time in LSL timebase

        Returns:
        float: Current time in seconds (LSL timebase)
        """
        from pylsl import local_clock
        return local_clock()
    
    def get_timing_options(self, r_peak_time, current_block):
        """Generate timing option based on the current block's condition using LSL timebase

        Parameters:
        r_peak_time (float): R-peak timestamp in seconds (LSL timebase)
        current_block (int): Current block number

        Returns:
        float: Target presentation time in seconds (LSL timebase)
        """
        timing_ms, phase = self.get_block_timing(current_block)
        # Explicit conversion from milliseconds to seconds for LSL timebase
        timing_seconds = timing_ms / 1000.0
        return r_peak_time + timing_seconds


    def save_symbol_allocations(self, all_block_outcomes, participant_info, reversal_points):
        """Save symbol allocations for all blocks with reversal points and win/loss percentages."""
        filename = f"{participant_info['participant']}-ses{participant_info['session']}-run{participant_info['run']}-{participant_info['date_time']}_allocations.txt"
        filepath = self.data_path / filename

        with open(filepath, 'w') as f:
            f.write("EXPERIMENT SYMBOL ALLOCATIONS\n")
            f.write("="*50 + "\n\n")

            # Ensure we process exactly n_blocks
            n_blocks = participant_info['n_blocks']  # Should be 6
            n_trials = participant_info['n_trials']  # Should be 10

            for block_idx in range(n_blocks):
                block_outcomes = all_block_outcomes[block_idx]
                reversal_point = reversal_points[block_idx]

                # Header for block
                f.write(f"BLOCK {block_idx + 1} REPORT\n")
                f.write(f"Reversal Point: Trial {reversal_point}\n")
                f.write("-"*50 + "\n\n")

                # Before Reversal
                f.write("Before Reversal:\n")
                f.write("-"*30 + "\n")
                before_reversal_a = block_outcomes['first_half'][0][:reversal_point]
                before_reversal_b = block_outcomes['first_half'][1][:reversal_point]

                for trial in range(reversal_point):
                    f.write(f"Trial {trial:2d}: Symbol A -> {'Win ' if before_reversal_a[trial] else 'Loss'} | ")
                    f.write(f"Symbol B -> {'Win ' if before_reversal_b[trial] else 'Loss'}\n")

                # Summary for Before Reversal
                before_a_wins = sum(before_reversal_a)
                before_b_wins = sum(before_reversal_b)
                f.write("\nBefore Reversal Summary:\n")
                f.write(f"Symbol A: {before_a_wins} Wins ({(before_a_wins/reversal_point)*100:.1f}%)\n")
                f.write(f"Symbol B: {before_b_wins} Wins ({(before_b_wins/reversal_point)*100:.1f}%)\n\n")

                # After Reversal
                f.write("After Reversal:\n")
                f.write("-"*30 + "\n")
                remaining_trials = n_trials - reversal_point
                after_reversal_a = block_outcomes['second_half'][0][:remaining_trials]
                after_reversal_b = block_outcomes['second_half'][1][:remaining_trials]

                for trial_offset in range(remaining_trials):
                    trial = trial_offset + reversal_point
                    f.write(f"Trial {trial:2d}: Symbol A -> {'Win ' if after_reversal_a[trial_offset] else 'Loss'} | ")
                    f.write(f"Symbol B -> {'Win ' if after_reversal_b[trial_offset] else 'Loss'}\n")

                # Summary for After Reversal
                after_a_wins = sum(after_reversal_a)
                after_b_wins = sum(after_reversal_b)
                f.write("\nAfter Reversal Summary:\n")
                f.write(f"Symbol A: {after_a_wins} Wins ({(after_a_wins/remaining_trials)*100:.1f}%)\n")
                f.write(f"Symbol B: {after_b_wins} Wins ({(after_b_wins/remaining_trials)*100:.1f}%)\n\n")

                f.write("="*50 + "\n\n")

    def print_trial_report(self, trial_data, trial_num, is_reversed):
        """Print a detailed report of the trial results with enhanced timing information"""
        print("\n" + "="*50)
        print(f"Trial {trial_num} Report (Reversed: {is_reversed})")
        if 'reversal_point' in trial_data:
            print(f"Reversal Point: Trial {trial_data['reversal_point']}")
        print("-"*50)

        if trial_data['choice'] is not None:
            print(f"Chosen Symbol: {trial_data['chosen_symbol']}")
            print(f"Symbol Positions: {trial_data['symbol_positions']}")
            print(f"Response Time: {trial_data['rt']:.3f}s")
            print(f"Feedback: {trial_data['feedback'].upper()}")
            print(f"Block Condition: Block {trial_data['block_condition']}")
            print(f"Cardiac Phase: {trial_data['cardiac_phase']}")
            print(f"Intended Delay: {trial_data['timing_ms']}ms")
            print(f"Actual Timing Precision: {trial_data['timing_precision_ms']:.2f}ms")
        else:
            print("Response: No response (Too slow)")
            print(f"Block Condition: Block {trial_data['block_condition']}")
            print(f"Cardiac Phase: {trial_data['cardiac_phase']}")
            print(f"Timing Precision: {trial_data['timing_precision_ms']:.2f}ms")

        print("="*50 + "\n")

    def save_block_order(self, participant_info):
        """Save the randomized block order information"""
        filename = f"{participant_info['participant']}-ses{participant_info['session']}-run{participant_info['run']}-{participant_info['date_time']}_block_order.txt"
        filepath = self.data_path / filename

        with open(filepath, 'w') as f:
            f.write("Block Order Information\n")
            f.write("=====================\n\n")
            for actual_block, info in self.block_timing_map.items():
                f.write(f"Block {actual_block}:\n")
                f.write(f"  Original Condition: Block {info['original_block']}\n")
                f.write(f"  Cardiac Phase: {info['phase']}\n")
                f.write(f"  Timing: {info['timing']}ms post R-peak\n\n")
    
    def cleanup(self):
        """Comprehensive cleanup of all resources with improved error handling"""
        print("\nInitiating cleanup sequence...")

        # First, try to save any pending data
        try:
            if hasattr(self, 'participant_info') and len(self.all_data) > 0:
                print("Saving experimental data before cleanup...")
                df = pd.DataFrame(self.all_data)
                filename = f"{self.participant_info['participant']}-ses{self.participant_info['session']}-run{self.participant_info['run']}-{self.participant_info['date_time']}.csv"
                df.to_csv(self.data_path / filename, index=False)
                print(f"Data saved as: {filename}")
        except Exception as e:
            print(f"Warning: Could not save data during cleanup: {e}")

        # Clean up Cedrus with verification
        try:
            if hasattr(self, 'cedrus_box') and self.cedrus_box:
                print("Cleaning up Cedrus device...")
                self.cedrus_box.clear_response_queue()
                time.sleep(0.5)  # Give device time to clear
                print("Cedrus device cleanup completed")
        except Exception as e:
            print(f"Warning: Cedrus cleanup error: {e}")

        # Clean up LSL
        try:
            if hasattr(self, 'inlet'):
                print("Closing LSL stream...")
                self.inlet.close_stream()
                print("LSL stream closed")
        except Exception as e:
            print(f"Warning: LSL cleanup error: {e}")

        # Clean up window
        try:
            if hasattr(self, 'win'):
                print("Closing PsychoPy window...")
                self.win.close()
                print("Window closed")
        except Exception as e:
            print(f"Warning: Window cleanup error: {e}")

        print("Cleanup completed")
    
    
    def run_trial(self, trial_num, n_trials, is_reversed, block_outcomes, current_block, reversal_point):
        """Run a single trial with optimized timing and improved error handling"""
        # Cache time correction at start of trial
        time_correction = self.inlet.time_correction()
        trial_start_time = self.get_lsl_time()

        self.send_marker(self.markers['trial_start'])
        self.cedrus_box.clear_response_queue()

        # Present fixation cross with shorter ITI
        iti_duration = np.random.uniform(self.settings['iti_min'], self.settings['iti_max'])
        fixation_timer = core.Clock()
        self.stim['fixation'].draw()
        self.win.flip()
        while fixation_timer.getTime() < iti_duration:
            self.cedrus_box.poll_for_response()
            core.wait(0.001)

        self.cedrus_box.clear_response_queue()

        # Decision phase setup
        left_pos, right_pos = (-0.15, 0), (0.15, 0)
        should_swap = np.random.random() < 0.5
        if should_swap:
            self.stim['symbols'][0].pos = right_pos  # Symbol A
            self.stim['symbols'][1].pos = left_pos   # Symbol B
            symbol_indices = [1, 0]  # Left position first
            symbol_order = 'BA'
        else:
            self.stim['symbols'][0].pos = left_pos   # Symbol A
            self.stim['symbols'][1].pos = right_pos  # Symbol B
            symbol_indices = [0, 1]  # Left position first
            symbol_order = 'AB'

        # Draw choice stimuli
        self.stim['fixation'].draw()
        self.stim['symbols'][0].draw()
        self.stim['symbols'][1].draw()

        # Start decision phase
        self.win.flip()
        decision_timer = core.Clock()

        # Initialize variables
        response_made = False
        response_key = None
        rt = None
        choice = None
        chosen_symbol = None
        feedback_type = 'neutral'
        actual_presentation = None
        r_peak_time = None
        feedback_time = None
        timing_ms = None
        phase = None

        # Response collection
        while decision_timer.getTime() < self.settings['decision_duration'] and not response_made:
            self.cedrus_box.poll_for_response()
            if self.cedrus_box.response_queue:
                response = self.cedrus_box.get_next_response()
                if response['pressed']:
                    if response['key'] in [0, 1]:
                        response_made = True
                        response_key = 'left' if response['key'] == 0 else 'right'
                        rt = decision_timer.getTime()
                    elif response['key'] == 6:  # Blue button for quit
                        print("\nBlue button pressed - initiating experiment termination...")
                        self.cleanup()
                        core.quit()

            core.wait(0.001)

        self.cedrus_box.clear_response_queue()
        timing_ms, phase = self.get_block_timing(current_block)

        # Clear R-peak queue before collecting new peak
        while not self.r_peak_times.empty():
            _ = self.r_peak_times.get()

        if response_made:
            self.send_marker(self.markers['choice_made'])

            # Calculate choice and outcome immediately
            choice_index = 0 if response_key == 'left' else 1
            choice = choice_index
            chosen_symbol = symbol_indices[choice_index]

            # Calculate feedback
            half = 'second_half' if is_reversed else 'first_half'
            trial_index = (trial_num - reversal_point if is_reversed else trial_num)
            trial_index = min(trial_index, len(block_outcomes[half][0]) - 1)
            outcome = block_outcomes[half][chosen_symbol][trial_index]
            feedback_type = 'win' if outcome else 'loss'

            # Show brief fixation
            self.stim['fixation'].draw()
            self.win.flip()
            core.wait(0.5)  # Fixed shorter delay

            # Get R-peak with timeout
            try:
                r_peak_time = self.r_peak_times.get(timeout=2)
            except queue.Empty:
                print("Warning: R-peak detection timeout")
                r_peak_time = self.get_lsl_time()

            # Prepare feedback presentation
            feedback_time = r_peak_time + (timing_ms / 1000.0)
            self.stim['feedback'][feedback_type].draw()

            # Efficient waiting with shorter window
            self.wait_for_timing(feedback_time, window=0.002)

            # Present feedback with precise timing
            pre_flip_time = self.get_lsl_time()
            self.send_marker(self.markers['feedback_onset'])
            feedback_marker = {
                'win': self.markers['win_feedback'],
                'loss': self.markers['loss_feedback']
            }[feedback_type]
            self.send_marker(feedback_marker)

            self.win.flip()
            post_flip_time = self.get_lsl_time()
            actual_presentation = post_flip_time  # More accurate timing

            core.wait(1.0)  # Shorter feedback duration

        else:
            # Timeout handling with shorter delays
            self.send_marker(self.markers['timeout'])

            try:
                r_peak_time = self.r_peak_times.get(timeout=2)
            except queue.Empty:
                print("Warning: R-peak detection timeout")
                r_peak_time = self.get_lsl_time()

            feedback_time = r_peak_time + (timing_ms / 1000.0)
            self.stim['timeout_msg'].draw()

            self.wait_for_timing(feedback_time, window=0.002)

            pre_flip_time = self.get_lsl_time()
            self.win.flip()
            post_flip_time = self.get_lsl_time()
            actual_presentation = post_flip_time

            core.wait(0.5)  # Shorter timeout display

        self.send_marker(self.markers['trial_end'])

        # Compile trial data
        trial_data = {
            'rt': rt,  # in seconds
            'choice': choice,
            'chosen_symbol': 'A' if chosen_symbol == 0 else ('B' if chosen_symbol == 1 else None),
            'symbol_positions': symbol_order,
            'feedback': feedback_type,
            'is_reversed': is_reversed,
            'block_condition': self.block_timing_map[current_block + 1]['original_block'],
            'cardiac_phase': phase,
            'timing_ms': timing_ms,
            'r_peak_time': r_peak_time,
            'intended_presentation_time': feedback_time,
            'actual_presentation_time': actual_presentation,
            'timing_precision_ms': (actual_presentation - feedback_time) * 1000.0,
            'response_time': trial_start_time,
            'reversal_point': reversal_point
        }

        if trial_data['timing_precision_ms'] is not None:
            if abs(trial_data['timing_precision_ms']) > 50:
                print(f"Warning: Large timing discrepancy detected: {trial_data['timing_precision_ms']:.2f}ms")

        self.print_trial_report(trial_data, trial_num, is_reversed)
        self.cedrus_box.clear_response_queue()

        return trial_data

    
    
    def run_experiment(self):
        """Run the complete experiment with markers"""
        # Initialize data storage before try block
        all_data = []
        participant_info = None
        reversal_points = []  # Initialize list to store reversal points

        try:
            # Setup
            participant_info = self.get_participant_info()
            self.setup_experiment()

            # Check Cedrus box before proceeding
            if self.cedrus_box is None:
                raise RuntimeError("Cedrus device not found or not properly initialized. Please check the connection and restart.")

            # Test Cedrus connection
            try:
                self.cedrus_box.poll_for_response()
                print("Cedrus device connection verified")
            except Exception as e:
                raise RuntimeError(f"Cedrus device test failed: {e}")

            # Send experiment start marker
            self.send_marker(self.markers['experiment_start'])

            # Setup LSL with better error handling
            try:
                self.setup_lsl()
                print("Waiting for R-peak stream...")
                core.wait(3.0)
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please ensure that the R-peak stream is running and try again.")
                raise  # Re-raise the exception to trigger cleanup

            # Show instructions
            self.show_instructions()

            # Initialize block outcomes storage
            all_block_outcomes = []

            # Run blocks
            for block in range(participant_info['n_blocks']):
                # Send block start marker
                self.send_marker(self.markers['block_start'])
                print(f"\nStarting Block {block + 1}...")

                # Calculate the midpoint of trials
                n_trials = participant_info['n_trials']
                midpoint = n_trials // 2

                # Define range for random reversal (±2 trials from midpoint)
                min_reversal = max(0, midpoint - 2)  # Ensure we don't go below 0
                max_reversal = min(n_trials, midpoint + 2)  # Ensure we don't exceed total trials

                # Generate random reversal point
                reversal_point = np.random.randint(min_reversal, max_reversal + 1)
                reversal_points.append(reversal_point)  # Store the reversal point for this block
                print(f"Randomized reversal point for Block {block + 1}: Trial {reversal_point}")

                # Initialize block outcomes
                block_outcomes = self.initialize_block_outcomes(n_trials, reversal_point)
                all_block_outcomes.append(block_outcomes)

                # Run trials
                for trial in range(n_trials):
                    # Check if we've reached the reversal point
                    is_reversed = trial >= reversal_point
                    if trial == reversal_point:
                        self.send_marker(self.markers['reversal'])
                        print("\nProbability Reversal!")

                    trial_data = self.run_trial(trial, n_trials, is_reversed, block_outcomes, block, reversal_point)
                    trial_data.update({
                        'block': block,
                        'trial': trial,
                        'reversal_point': reversal_point,  # Save the randomized reversal point
                        'participant': participant_info['participant'],
                        'session': participant_info['session'],
                        'run': participant_info['run'],
                        'date_time': participant_info['date_time']
                    })
                    all_data.append(trial_data)

                # Send block end marker
                self.send_marker(self.markers['block_end'])
                print(f"\nBlock {block + 1} completed.")

                # Show the break message if not the last block
                if block < participant_info['n_blocks'] - 1:
                    self.show_break_message()

            # After ALL blocks are completed, save symbol allocations with reversal points
            self.save_symbol_allocations(all_block_outcomes, participant_info, reversal_points)

        except Exception as e:
            print(f"Experiment error: {e}")
        finally:
            # Ensure cleanup happens regardless of how we exit
            self.cleanup_cedrus()
            self.send_marker(self.markers['experiment_end'])

            # Only save data if we have both data and participant info
            if all_data and participant_info:
                try:
                    df = pd.DataFrame(all_data)

                    # Add timing column verification here
                    timing_columns = ['r_peak_time', 'intended_presentation_time', 
                                    'actual_presentation_time', 'timing_precision_ms']
                    for col in timing_columns:
                        if col not in df.columns:
                            print(f"Warning: {col} not found in trial data")

                    # Create filename and save data
                    filename = f"{participant_info['participant']}-ses{participant_info['session']}-run{participant_info['run']}-{participant_info['date_time']}.csv"
                    df.to_csv(self.data_path / filename, index=False)
                    print(f"\nData saved as: {filename}")
                except Exception as e:
                    print(f"Error saving data: {e}")

            if hasattr(self, 'win'):
                self.win.close()

        # After all blocks are completed
        self.save_block_order(participant_info)

if __name__ == "__main__":
    task = CardiacSyncedLearningTask()
    task.run_experiment()