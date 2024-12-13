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
            'decision_duration': 1.25,  # Decision phase duration
            'delay_min': 1.5,  # Minimum delay duration
            'delay_max': 2.0,  # Maximum delay duration
            'feedback_duration': 4.0,  # Feedback duration
            'iti_min': 1.0,  # Minimum inter-trial interval
            'iti_max': 2.0,  # Maximum inter-trial interval
            'win_probability_good': 0.70,  # Probability of win for good symbol
            'win_probability_bad': 0.30,  # Probability of win for bad symbol
            'systole_window_start': 0,      # ms after R-peak
            'systole_window_end': 300,      # ms after R-peak
            'diastole_window_start': 300,   # ms after R-peak
            'diastole_window_end': 700,     # ms after R-peak
            'grid_resolution': 100           # ms steps
        }

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

        # Task instructions [previous instructions remain the same]
        self.instructions = [
            """Welcome to the experiment!

    In this task, you will see two symbols on each trial.
    Your goal is to learn which symbol is more likely to give you rewards.
    Note that these probabilities may change during the task.

    Press YELLOW BUTTON to continue...""",

            """On each trial:
    1. Two arrows will appear on the screen
    2. Choose the left arrow with the LEFT ARROW key
    3. Choose the right arrow with the RIGHT ARROW key
    4. You have 1.25 seconds to make your choice

    Press YELLOW BUTTON to continue...""",

            """After your choice:
    - If you see an UPWARD ARROW, your choice was CORRECT
    - If you see a DOWNWARD ARROW, your choice was INCORRECT
    - The same symbols will have different probabilities of being correct
    - These probabilities will change during each block

    Press YELLOW BUTTON to continue...""",

            """Important Notes:
    - Respond as quickly and accurately as possible
    - If you don't respond in time, you'll see a warning message
    - You can press 'BLUE KEY' at any time to end the experiment
    - Your data will be saved automatically

    Press YELLOW BUTTON to begin the task."""
        ]

        # Initialize queue for R-peak times
        self.r_peak_times = queue.Queue()

    def cleanup_cedrus(self):
        """Safely cleanup Cedrus device connection"""
        try:
            if hasattr(self, 'cedrus_box') and self.cedrus_box:
                self.cedrus_box.clear_response_queue()
                time.sleep(0.5)  # Give device time to clear
        except Exception as e:
            print(f"Error during Cedrus cleanup: {e}")

    def generate_fixed_outcomes(self, n_trials, win_probability):
        """Generate a fixed sequence of outcomes that exactly matches the desired probability"""
        n_wins = round(n_trials * win_probability)
        outcomes = ([True] * n_wins) + ([False] * (n_trials - n_wins))
        np.random.shuffle(outcomes)
        return outcomes

    def initialize_block_outcomes(self, n_trials):
        """Initialize predetermined outcomes for a block"""
        half_trials = n_trials // 2

        # Generate outcomes for first half
        symbol_a_first = self.generate_fixed_outcomes(half_trials, self.settings['win_probability_good'])
        symbol_b_first = self.generate_fixed_outcomes(half_trials, self.settings['win_probability_bad'])

        # Generate outcomes for second half (reversed probabilities)
        symbol_a_second = self.generate_fixed_outcomes(n_trials - half_trials, self.settings['win_probability_bad'])
        symbol_b_second = self.generate_fixed_outcomes(n_trials - half_trials, self.settings['win_probability_good'])

        return {
            'first_half': {
                0: symbol_a_first,
                1: symbol_b_first
            },
            'second_half': {
                0: symbol_a_second,
                1: symbol_b_second
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
                color='red'
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
        """Setup LSL inlet for R-peak markers"""
        print("Looking for R-peak markers stream...")
        try:
            streams = resolve_stream('type', 'R_PEAK')
            if not streams:
                raise RuntimeError("No R-peak stream found")
            self.inlet = StreamInlet(streams[0])
            print("R-peak stream found and connected!")

            # Start R-peak collection thread
            self.lsl_thread = threading.Thread(target=self.collect_r_peaks)
            self.lsl_thread.daemon = True
            self.lsl_thread.start()
            print("R-peak collection thread started")

        except Exception as e:
            raise RuntimeError(f"Failed to setup LSL stream: {str(e)}")

    def collect_r_peaks(self):
        """Continuously collect R-peak markers"""
        while True:
            sample, timestamp = self.inlet.pull_sample()
            self.r_peak_times.put(timestamp)

    def get_participant_info(self):
        """Show dialog to collect participant information"""
        current_time = time.strftime("%Y%m%d-%H%M%S")

        exp_info = {
            'participant': '',
            'session': '001',
            'run': '1',
            'n_blocks': '6',
            'n_trials': '40',
            'date_time': current_time,
        }

        dlg = gui.DlgFromDict(
            dictionary=exp_info,
            title='Task Info',
            fixed=['date_time']
        )

        if dlg.OK:
            # Convert numeric fields to integers
            exp_info['n_blocks'] = int(exp_info['n_blocks'])
            exp_info['n_trials'] = int(exp_info['n_trials'])
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

    def get_timing_mode(self, trial_num, n_trials):
        """Determine timing mode based on trial position within block"""
        third = n_trials // 3
        if trial_num < third:
            return 'systole'
        elif trial_num < 2 * third:
            return 'diastole'
        else:
            return 'mixed'

    def get_timing_options(self, r_peak_time):
        """Generate exactly three timing options for systole and diastole phases,
        with the first and last points shifted 10 ms inside the boundaries and the middle point at the center."""
        options = {'systole': [], 'diastole': []}

        # Calculate systole points
        systole_start = self.settings['systole_window_start'] + 10
        systole_end = self.settings['systole_window_end'] - 10
        systole_middle = (self.settings['systole_window_start'] + self.settings['systole_window_end']) / 2

        systole_times = [systole_start, systole_middle, systole_end]

        # Generate systole points
        for t in systole_times:
            options['systole'].append(r_peak_time + t / 1000)  # Convert ms to seconds

        # Calculate diastole points
        diastole_start = self.settings['diastole_window_start'] + 10
        diastole_end = self.settings['diastole_window_end'] - 10
        diastole_middle = (self.settings['diastole_window_start'] + self.settings['diastole_window_end']) / 2

        diastole_times = [diastole_start, diastole_middle, diastole_end]

        # Generate diastole points
        for t in diastole_times:
            options['diastole'].append(r_peak_time + t / 1000)  # Convert ms to seconds

        #print("Systole points:", options['systole'])
        #print("Diastole points:", options['diastole'])

        return options


    def select_feedback_time(self, timing_mode, timing_options):
        """Select feedback presentation time based on mode"""
        if timing_mode == 'systole':
            options = timing_options['systole']
        elif timing_mode == 'diastole':
            options = timing_options['diastole']
        else:  # mixed
            options = timing_options['systole'] + timing_options['diastole']

        return np.random.choice(options) if options else time.time()

    def save_symbol_allocations(self, block_outcomes, participant_info):
        """Save symbol allocations for each block"""
        filename = f"{participant_info['participant']}-ses{participant_info['session']}-run{participant_info['run']}-{participant_info['date_time']}_allocations.txt"
        filepath = self.data_path / filename

        with open(filepath, 'w') as f:
            half_trials = len(block_outcomes['first_half'][0])
            total_trials = half_trials * 2

            # Write block information
            f.write(f"\nBLOCK OUTCOMES\n")
            f.write("="*50 + "\n\n")

            # First half
            f.write("First Half (Pre-reversal):\n")
            f.write("-"*30 + "\n")

            first_half_a = block_outcomes['first_half'][0]
            first_half_b = block_outcomes['first_half'][1]

            for trial in range(half_trials):
                f.write(f"Trial {trial:2d}: ")
                f.write(f"Symbol A -> {'Win ' if first_half_a[trial] else 'Loss'} | ")
                f.write(f"Symbol B -> {'Win ' if first_half_b[trial] else 'Loss'}\n")

            # Calculate statistics
            first_half_a_wins = sum(first_half_a)
            first_half_b_wins = sum(first_half_b)

            f.write("\nFirst Half Summary:\n")
            f.write(f"Symbol A: {first_half_a_wins} Wins ({(first_half_a_wins/half_trials)*100:.1f}%)\n")
            f.write(f"Symbol B: {first_half_b_wins} Wins ({(first_half_b_wins/half_trials)*100:.1f}%)\n")

            # Second half
            f.write("\nSecond Half (Post-reversal):\n")
            f.write("-"*30 + "\n")

            second_half_a = block_outcomes['second_half'][0]
            second_half_b = block_outcomes['second_half'][1]

            for trial in range(len(second_half_a)):
                trial_num = trial + half_trials
                f.write(f"Trial {trial_num:2d}: ")
                f.write(f"Symbol A -> {'Win ' if second_half_a[trial] else 'Loss'} | ")
                f.write(f"Symbol B -> {'Win ' if second_half_b[trial] else 'Loss'}\n")

            # Calculate second half statistics
            second_half_a_wins = sum(second_half_a)
            second_half_b_wins = sum(second_half_b)

            f.write("\nSecond Half Summary:\n")
            f.write(f"Symbol A: {second_half_a_wins} Wins ({(second_half_a_wins/len(second_half_a))*100:.1f}%)\n")
            f.write(f"Symbol B: {second_half_b_wins} Wins ({(second_half_b_wins/len(second_half_b))*100:.1f}%)\n")

    def print_trial_report(self, trial_data, trial_num, is_reversed):
        """Print a detailed report of the trial results"""
        print("\n" + "="*50)
        print(f"Trial {trial_num} Report (Reversed: {is_reversed})")
        print("-"*50)

        if trial_data['choice'] is not None:
            print(f"Chosen Symbol: {trial_data['chosen_symbol']}")
            print(f"Symbol Positions: {trial_data['symbol_positions']}")
            print(f"Response Time: {trial_data['rt']:.3f}s")
        else:
            print("Response: No response (Too slow)")

        print(f"Feedback: {trial_data['feedback'].upper()}")
        print(f"Timing Mode: {trial_data['timing_mode']}")
        print(f"Cardiac Phase: {trial_data['cardiac_phase']}")
        print(f"Time from R-peak: {trial_data['time_from_r_peak']*1000:.1f}ms")
        print("="*50 + "\n")

    def run_trial(self, trial_num, n_trials, is_reversed, block_outcomes):
        """Run a single trial with conservative cardiac-synced feedback"""
        # Determine timing mode
        timing_mode = self.get_timing_mode(trial_num, n_trials)

        # Send trial start marker
        self.send_marker(self.markers['trial_start'])

        # Clear any pending responses before trial starts
        self.cedrus_box.clear_response_queue()

        # Present fixation cross and ignore responses
        iti_duration = np.random.uniform(self.settings['iti_min'], self.settings['iti_max'])
        fixation_timer = core.Clock()
        self.stim['fixation'].draw()
        self.win.flip()
        while fixation_timer.getTime() < iti_duration:
            self.cedrus_box.poll_for_response()  # Ignore any responses silently
            core.wait(0.001)

        # Clear any residual responses
        self.cedrus_box.clear_response_queue()


        # Clear responses again before decision phase
        self.cedrus_box.clear_response_queue()

        # Decision Phase
        t_0 = time.time()  # Response time

        # Present symbols with fixation cross
        positions = [(-0.15, 0), (0.15, 0)]
        symbol_indices = [0, 1]

        # Randomize positions
        combined = list(zip(self.stim['symbols'], positions, symbol_indices))
        np.random.shuffle(combined)
        symbols, positions, indices = zip(*combined)

        # Draw fixation cross and symbols
        self.stim['fixation'].draw()
        for sym, pos in zip(symbols, positions):
            sym.pos = pos
            sym.draw()

        # Start decision phase
        self.win.flip()
        decision_timer = core.Clock()
        response_made = False
        response_key = None
        rt = None

        # Only collect responses during decision phase
        while decision_timer.getTime() < self.settings['decision_duration'] and not response_made:
            self.cedrus_box.poll_for_response()
            if self.cedrus_box.response_queue:
                response = self.cedrus_box.get_next_response()
                if response['pressed']:  # Only handle button press, not release
                    if response['key'] in [0, 1]:  # Only accept buttons 0 and 1
                        response_made = True
                        response_key = 'left' if response['key'] == 0 else 'right'
                        rt = decision_timer.getTime()
                    elif response['key'] == 6:  # Blue button to quit
                        self.cleanup_cedrus()
                        self.win.close()
                        core.quit()

            core.wait(0.001)  # Prevent CPU overload

        # Clear the response queue after decision phase
        self.cedrus_box.clear_response_queue()

        # Process response
        if not response_made:
            self.send_marker(self.markers['timeout'])
            self.stim['timeout_msg'].draw()
            self.win.flip()
            core.wait(1.0)
            feedback_type = 'neutral'
            rt = None
            choice = None
            chosen_symbol = None
        else:
            self.send_marker(self.markers['choice_made'])
            choice = 0 if response_key == 'left' else 1
            chosen_symbol = indices[choice]

            # After choice, show only fixation cross
            self.stim['fixation'].draw()
            self.win.flip()
            post_choice_delay = np.random.uniform(1.0, 2.0)
            core.wait(post_choice_delay)

            # Use predetermined outcome based on trial position and chosen symbol
            half = 'second_half' if is_reversed else 'first_half'
            trial_index = trial_num - (n_trials // 2) if is_reversed else trial_num
            outcome = block_outcomes[half][chosen_symbol][trial_index]
            feedback_type = 'win' if outcome else 'loss'

        # Clear responses before waiting for R-peaks
        self.cedrus_box.clear_response_queue()

        # Wait for TWO R-peaks after choice for conservative timing
        r_peak_1 = self.r_peak_times.get()  # First R-peak after choice
        r_peak_2 = self.r_peak_times.get()  # Second R-peak

        # Generate timing options for second R-peak
        timing_options = self.get_timing_options(r_peak_2)

        # Select feedback time
        feedback_time = self.select_feedback_time(timing_mode, timing_options)

        # Wait until selected feedback time
        while time.time() < feedback_time:
            core.wait(0.001)

        # Show feedback (only if there was a response)
        if choice is not None:
            feedback_marker = {
                'win': self.markers['win_feedback'],
                'loss': self.markers['loss_feedback']
            }[feedback_type]

            self.send_marker(self.markers['feedback_onset'])
            self.send_marker(feedback_marker)

            self.stim['feedback'][feedback_type].draw()
            self.win.flip()
            core.wait(self.settings['feedback_duration'])

        # Send trial end marker
        self.send_marker(self.markers['trial_end'])

        # Calculate timing information
        cardiac_phase = ('systole' if feedback_time - r_peak_2 <= 0.300 else 'diastole')

        # Create trial data
        trial_data = {
            'rt': rt,
            'choice': choice,
            'chosen_symbol': 'A' if chosen_symbol == 0 else 'B',
            'symbol_positions': 'AB' if indices[0] == 0 else 'BA',
            'feedback': feedback_type,
            'is_reversed': is_reversed,
            'timing_mode': timing_mode,
            'r_peak_time': r_peak_2,
            'feedback_time': feedback_time,
            'cardiac_phase': cardiac_phase,
            'time_from_r_peak': feedback_time - r_peak_2,
            'response_time': t_0,
            'first_r_peak': r_peak_1,
            'second_r_peak': r_peak_2
        }

        # Print trial report
        self.print_trial_report(trial_data, trial_num, is_reversed)

        # Clear responses before ending trial
        self.cedrus_box.clear_response_queue()

        return trial_data

    def run_experiment(self):
        """Run the complete experiment with markers"""
        # Initialize data storage before try block
        all_data = []
        participant_info = None

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

                # Show fixation cross before the first trial of the block
                iti_duration = np.random.uniform(self.settings['iti_min'], self.settings['iti_max'])
                fixation_timer = core.Clock()  # Start a timer for the fixation period
                self.stim['fixation'].draw()
                self.win.flip()

                while fixation_timer.getTime() < iti_duration:
                    self.cedrus_box.poll_for_response()  # Ignore responses silently
                    core.wait(0.001)  # Avoid CPU overload

                # Clear residual responses after the fixation cross
                self.cedrus_box.clear_response_queue()

                # Initialize block outcomes
                n_trials = participant_info['n_trials']
                reversal_point = n_trials // 2
                block_outcomes = self.initialize_block_outcomes(n_trials)
                all_block_outcomes.append(block_outcomes)

                # Run trials
                for trial in range(n_trials):
                    # Check if we've reached the reversal point
                    is_reversed = trial >= reversal_point
                    if trial == reversal_point:
                        self.send_marker(self.markers['reversal'])
                        print("\nProbability Reversal!")

                    trial_data = self.run_trial(trial, n_trials, is_reversed, block_outcomes)
                    trial_data.update({
                        'block': block,
                        'trial': trial,
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

            # After ALL blocks are completed, save symbol allocations
            for block, block_outcomes in enumerate(all_block_outcomes):
                self.save_symbol_allocations(block_outcomes, participant_info)

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
                    filename = f"{participant_info['participant']}-ses{participant_info['session']}-run{participant_info['run']}-{participant_info['date_time']}.csv"
                    df.to_csv(self.data_path / filename, index=False)
                    print(f"\nData saved as: {filename}")
                except Exception as e:
                    print(f"Error saving data: {e}")

            if hasattr(self, 'win'):
                self.win.close()


if __name__ == "__main__":
    task = CardiacSyncedLearningTask()
    task.run_experiment()