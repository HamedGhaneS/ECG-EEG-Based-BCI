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
        'decision_duration': 1.25,
        'delay_min': 1,
        'delay_max': 1.5,
        'feedback_duration': 4.0,
        'iti_min': 1.0,
        'iti_max': 1.5,
        'win_probability_good': 0.70,
        'win_probability_bad': 0.30,
        'rr_interval_min': 0.5,
        'rr_interval_max': 1.2,
        'monitor_refresh_delay': 0.009,
        'rpeak_queue_size': 20
        }
         
        self.current_rr_average = None  # Store running R-R average
        self.rr_window_size = 5        # Number of intervals to use for running average
        self.recent_rr_intervals = []   # Store recent intervals for running average
    
        self.settings.update({
                'rr_interval_min': 0.6,  # Minimum physiologically valid R-R interval in seconds
                'rr_interval_max': 1.2,  # Maximum physiologically valid R-R interval in seconds
                'monitor_refresh_delay': 0.009,  # 9ms for 120Hz monitor
                'rpeak_queue_size': 20,  # Increased from 10 for better R-R calculations
        })
        
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
        
        self.r_peaks_block = []  # Store R-peaks for the current block
        self.current_r_peak = None  # Store the most recent R-peak
    
    def update_rr_average(self, new_peak_time):
        """Continuously update running R-R interval average.

        This method maintains a sliding window of R-R intervals and updates
        the average whenever a new R-peak is detected. This ensures we always
        have an up-to-date estimate of the current heart rate without
        introducing delays during trial execution.

        Args:
            new_peak_time (float): Timestamp of the newly detected R-peak
        """
        if hasattr(self, 'last_peak_time') and self.last_peak_time is not None:
            # Calculate new interval
            interval = new_peak_time - self.last_peak_time

            # Validate interval
            if self.settings['rr_interval_min'] <= interval <= self.settings['rr_interval_max']:
                # Add to recent intervals
                self.recent_rr_intervals.append(interval)

                # Keep only the most recent intervals
                if len(self.recent_rr_intervals) > self.rr_window_size:
                    self.recent_rr_intervals.pop(0)

                # Update running average
                if len(self.recent_rr_intervals) >= 2:  # Need at least 2 intervals
                    self.current_rr_average = np.mean(self.recent_rr_intervals)
                    print(f"Updated R-R average: {self.current_rr_average:.3f}s "
                          f"({60/self.current_rr_average:.1f} BPM)")

        # Store current peak time for next interval calculation
        self.last_peak_time = new_peak_time
    
    
    

    def calculate_cardiac_timing_grid(self, ave_rr):
        """Calculate cardiac timing grid with validation.

        Args:
            ave_rr (float): Average R-R interval in seconds

        Returns:
            dict: Timing grid or None if invalid
        """
        if not ave_rr or not (self.settings['rr_interval_min'] <= ave_rr <= self.settings['rr_interval_max']):
            print(f"Warning: Invalid R-R interval ({ave_rr:.3f}s)")
            return None

        return {
            'systole': {
                'early': 0.06 * ave_rr,
                'mid': 0.21 * ave_rr,
                'late': 0.36 * ave_rr
            },
            'diastole': {
                'early': 0.50 * ave_rr,
                'mid': 0.70 * ave_rr,
                'late': 0.90 * ave_rr
            }
        }

    def prepare_feedback(self, feedback_type):
        """Prepare feedback stimulus in advance.

        Args:
            feedback_type (str): Type of feedback ('win' or 'loss')
        """
        self.stim['feedback'][feedback_type].draw()
    
    
    def create_randomized_block_order(self):
        """Create a mapping of block numbers to randomized cardiac phase timing conditions.

        This method creates a randomized order of six conditions:
        Systole:  early (6%), mid (21%), late (36%)
        Diastole: early (50%), mid (70%), late (90%)

        Each condition represents a specific percentage of the R-R interval for 
        feedback presentation timing.
        """
        # Define the six timing conditions with their R-R interval percentages
        timing_conditions = [
            ('systole', 'early', 0.06),   # Early systole: 6% of R-R
            ('systole', 'mid', 0.21),     # Mid systole: 21% of R-R
            ('systole', 'late', 0.36),    # Late systole: 36% of R-R
            ('diastole', 'early', 0.50),  # Early diastole: 50% of R-R
            ('diastole', 'mid', 0.70),    # Mid diastole: 70% of R-R
            ('diastole', 'late', 0.90)    # Late diastole: 90% of R-R
        ]

        # Shuffle the timing conditions
        np.random.shuffle(timing_conditions)

        # Create mapping of block numbers (1-6) to shuffled conditions
        timing_map = {}
        for block_num, (phase, timing, percentage) in enumerate(timing_conditions, 1):
            timing_map[block_num] = {
                'original_block': block_num,
                'phase': phase,
                'timing': timing,
                'percentage': percentage  # Store the R-R percentage for this condition
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
        """Continuously collect R-peak markers and update R-R average."""
        while True:
            try:
                sample, timestamp = self.inlet.pull_sample(timeout=0.1)
                if sample is not None:
                    # Update running R-R average
                    self.update_rr_average(timestamp)

                    # Store peak in queue for timing calculations
                    self.r_peak_times.put((time.time(), timestamp))

                    if self.r_peak_times.qsize() > self.settings['rpeak_queue_size']:
                        _ = self.r_peak_times.get()
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"R-peak collection warning: {e}")
                time.sleep(0.001)

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
        """Print a detailed report of the trial results with enhanced cardiac timing information"""
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
            print(f"Block Condition: {trial_data['cardiac_phase']}-{trial_data['block_condition']}")

            if trial_data['ave_rr'] is not None:
                print(f"Average R-R Interval: {trial_data['ave_rr']:.3f}s")

            if trial_data['percentage_rr'] is not None:
                print(f"Target R-R percentage: {trial_data['percentage_rr']*100:.1f}%")

            if trial_data['delay'] is not None:
                print(f"Cardiac Timing Delay: {trial_data['delay']*1000:.1f}ms")

            if trial_data['timing_precision_ms'] is not None:
                print(f"Timing Precision: {trial_data['timing_precision_ms']:.2f}ms")
        else:
            print("Response: No response (Too slow)")
            print(f"Block Condition: {trial_data['cardiac_phase']}-{trial_data['block_condition']}")

    print("="*50 + "\n")

    def save_block_order(self, participant_info):
        """Save the randomized block order information with cardiac timing details"""
        filename = f"{participant_info['participant']}-ses{participant_info['session']}-run{participant_info['run']}-{participant_info['date_time']}_block_order.txt"
        filepath = self.data_path / filename

        with open(filepath, 'w') as f:
            f.write("Block Order Information\n")
            f.write("=====================\n\n")
            for actual_block, info in self.block_timing_map.items():
                f.write(f"Block {actual_block}:\n")
                f.write(f"  Original Condition: Block {info['original_block']}\n")
                f.write(f"  Cardiac Phase: {info['phase']}\n")
                f.write(f"  Timing Point: {info['timing']}\n")
                f.write(f"  R-R Percentage: {info['percentage']*100:.1f}%\n")
                f.write("-"*50 + "\n\n")
    
    def cleanup(self):
        """Comprehensive cleanup of all resources with error handling"""
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
        """Run a single trial with comprehensive timing measurements and cardiac synchronization.

        This method captures detailed timing information for each phase of the trial:
        - Trial start
        - Symbols presentation
        - Choice timing (if made)
        - Post-choice fixation
        - Outcome presentation
        - Trial end

        Each timing point is recorded using the LSL timebase for synchronization with 
        physiological data.

        Args:
            trial_num (int): Current trial number
            n_trials (int): Total number of trials in block
            is_reversed (bool): Whether probability reversal has occurred
            block_outcomes (dict): Predetermined outcomes for the block
            current_block (int): Current block number
            reversal_point (int): Trial number where probability reversal occurs
        """
        # Initialize all timing variables at the start of trial
        trial_timestamps = {
            'trial_start': None,           # When trial begins
            'symbols_onset': None,         # When choice symbols appear
            'choice_made': None,           # When participant makes a choice
            'fixation_post_choice': None,  # When fixation appears after choice
            'outcome_onset': None,         # When feedback appears
            'trial_end': None              # When trial ends
        }

        # Initialize cardiac timing variables
        phase = None
        delay = None
        ave_rr = None
        intended_presentation_time = None
        actual_presentation = None

        # Record trial start and send marker
        trial_timestamps['trial_start'] = self.get_lsl_time()
        self.send_marker(self.markers['trial_start'])
        self.cedrus_box.clear_response_queue()

        # Present fixation cross with random ITI
        iti_duration = np.random.uniform(self.settings['iti_min'], self.settings['iti_max'])
        fixation_timer = core.Clock()
        self.stim['fixation'].draw()
        self.win.flip()
        while fixation_timer.getTime() < iti_duration:
            self.cedrus_box.poll_for_response()
            core.wait(0.001)

        # Set up decision phase
        self.cedrus_box.clear_response_queue()
        left_pos, right_pos = (-0.15, 0), (0.15, 0)
        should_swap = np.random.random() < 0.5

        # Randomize symbol positions
        if should_swap:
            self.stim['symbols'][0].pos = right_pos
            self.stim['symbols'][1].pos = left_pos
            symbol_indices = [1, 0]
            symbol_order = 'BA'
        else:
            self.stim['symbols'][0].pos = left_pos
            self.stim['symbols'][1].pos = right_pos
            symbol_indices = [0, 1]
            symbol_order = 'AB'

        # Present choice stimuli and record onset time
        self.stim['fixation'].draw()
        self.stim['symbols'][0].draw()
        self.stim['symbols'][1].draw()
        self.win.flip()
        trial_timestamps['symbols_onset'] = self.get_lsl_time()
        decision_timer = core.Clock()

        # Initialize response variables
        response_made = False
        response_key = None
        rt = None
        choice = None
        chosen_symbol = None
        feedback_type = 'neutral'

        # Collect response during decision window
        while decision_timer.getTime() < self.settings['decision_duration'] and not response_made:
            self.cedrus_box.poll_for_response()
            if self.cedrus_box.response_queue:
                response = self.cedrus_box.get_next_response()
                if response['pressed']:
                    if response['key'] in [0, 1]:  # Left or Right response
                        response_made = True
                        response_key = 'left' if response['key'] == 0 else 'right'
                        rt = decision_timer.getTime()
                        trial_timestamps['choice_made'] = self.get_lsl_time()
                        # Log detailed timing information
                        print("\nChoice made:")
                        print(f"Choice LSL time: {trial_timestamps['choice_made']:.3f}")
                        print(f"Reaction time: {rt:.3f}s")
                        print(f"Choice direction: {response_key}\n")
                    elif response['key'] == 6:  # Quit response
                        self.cleanup()
                        core.quit()
            core.wait(0.001)

        if response_made:
            self.send_marker(self.markers['choice_made'])
            post_decision_timer = core.Clock()

            # Calculate choice outcome
            choice_index = 0 if response_key == 'left' else 1
            choice = choice_index
            chosen_symbol = symbol_indices[choice_index]
            half = 'second_half' if is_reversed else 'first_half'
            trial_index = (trial_num - reversal_point if is_reversed else trial_num)
            trial_index = min(trial_index, len(block_outcomes[half][0]) - 1)
            outcome = block_outcomes[half][chosen_symbol][trial_index]
            feedback_type = 'win' if outcome else 'loss'

            # Show fixation cross and record time
            self.stim['fixation'].draw()
            self.win.flip()
            trial_timestamps['fixation_post_choice'] = self.get_lsl_time()

            try:
                print("Starting cardiac timing sequence...")

                if self.current_rr_average is not None:
                    ave_rr = self.current_rr_average
                    print(f"Using current R-R average: {ave_rr:.3f}s")

                    # Get timing parameters
                    block_info = self.block_timing_map[current_block + 1]
                    phase = block_info['phase']
                    timing_point = block_info['timing']
                    rr_percentage = block_info['percentage']
                    delay = ave_rr * rr_percentage

                    print(f"Block condition: {phase}-{timing_point}")
                    print(f"Target timing: {rr_percentage*100:.1f}% of R-R")
                    print(f"Calculated delay: {delay*1000:.1f}ms")

                    # Prepare feedback stimulus
                    self.prepare_feedback(feedback_type)

                    # First wait for minimum post-decision time
                    while post_decision_timer.getTime() < self.settings['delay_min']:
                        core.wait(0.001)

                    # Then wait for next R-peak and calculate timing
                    got_timing = False
                    timeout_start = time.time()

                    while not got_timing and (time.time() - timeout_start) < 5.0:
                        if not self.r_peak_times.empty():
                            _, peak_time = self.r_peak_times.get()
                            intended_presentation_time = peak_time + delay - self.settings['monitor_refresh_delay']

                            if self.get_lsl_time() < intended_presentation_time:
                                print(f"R-peak detected at: {peak_time:.6f}")
                                print(f"Intended presentation time: {intended_presentation_time:.6f}")
                                got_timing = True
                                break

                        core.wait(0.001)

                    if got_timing:
                        # Wait for precise timing
                        self.wait_for_timing(intended_presentation_time, window=0.002)

                        # Present feedback
                        pre_flip_time = self.get_lsl_time()
                        self.send_marker(self.markers['feedback_onset'])
                        self.send_marker(self.markers[f'{feedback_type}_feedback'])
                        self.win.flip()
                        actual_presentation = self.get_lsl_time()
                        trial_timestamps['outcome_onset'] = actual_presentation
                        print(f"Actual presentation time: {actual_presentation:.6f}")

                        # Show feedback for specified duration
                        core.wait(self.settings['feedback_duration'])
                    else:
                        print("Warning: Failed to find suitable R-peak timing")

            except Exception as e:
                print(f"Error in cardiac timing sequence: {str(e)}")

        else:
            # Handle timeout case
            self.send_marker(self.markers['timeout'])
            self.stim['timeout_msg'].draw()
            self.win.flip()
            core.wait(0.5)

        # Record trial end time
        trial_timestamps['trial_end'] = self.get_lsl_time()
        self.send_marker(self.markers['trial_end'])

        # Compile trial data with all timing information
        trial_data = {
            'trial_start_time': trial_timestamps['trial_start'],
            'symbols_onset_time': trial_timestamps['symbols_onset'],
            'choice_time': trial_timestamps['choice_made'],
            'post_choice_fixation_time': trial_timestamps['fixation_post_choice'],
            'outcome_onset_time': trial_timestamps['outcome_onset'],
            'trial_end_time': trial_timestamps['trial_end'],
            'rt': rt,
            'choice': choice,
            'chosen_symbol': 'A' if chosen_symbol == 0 else ('B' if chosen_symbol == 1 else None),
            'symbol_positions': symbol_order,
            'feedback': feedback_type,
            'is_reversed': is_reversed,
            'block_condition': self.block_timing_map[current_block + 1]['timing'],
            'cardiac_phase': phase,
            'ave_rr': ave_rr,
            'delay': delay,
            'intended_presentation_time': intended_presentation_time,
            'actual_presentation_time': actual_presentation,
            'timing_precision_ms': (actual_presentation - intended_presentation_time) * 1000.0 if (actual_presentation is not None and intended_presentation_time is not None) else None,
            'percentage_rr': rr_percentage if 'rr_percentage' in locals() else None
        }

        # Print trial report and clear response queue
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
                
                # Clear R-peaks list at the start of each block
                self.r_peaks_block = []
                
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
