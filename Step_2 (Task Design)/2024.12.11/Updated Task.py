from psychopy import visual, core, data, event, gui
import pandas as pd
import numpy as np
from pathlib import Path
from pylsl import StreamInlet, resolve_stream, StreamInfo, StreamOutlet
import threading
import queue
import time

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
            'grid_resolution': 10           # ms steps
        }

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

        # Marker codes
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
            'neutral_feedback': 'feedback_neutral'
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

Press SPACE to continue...""",

        """On each trial:
1. Two arrows will appear on the screen
2. Choose the left arrow with the LEFT ARROW key
3. Choose the right arrow with the RIGHT ARROW key
4. You have 1.25 seconds to make your choice

Press SPACE to continue...""",

        """After your choice:
- If you see an UPWARD ARROW, your choice was CORRECT
- If you see a DOWNWARD ARROW, your choice was INCORRECT
- The same symbols will have different probabilities of being correct
- These probabilities will change during each block

Press SPACE to continue...""",

        """Important Notes:
- Respond as quickly and accurately as possible
- If you don't respond in time, you'll see a warning message
- You can press 'q' at any time to end the experiment
- Your data will be saved automatically

Press SPACE to begin the task."""
        ]

        # Initialize queue for R-peak times
        self.r_peak_times = queue.Queue()

    def generate_fixed_outcomes(self, n_trials, win_probability):
        """
        Generate a fixed sequence of outcomes that exactly matches the desired probability
        """
        # Calculate exact number of wins needed
        n_wins = round(n_trials * win_probability)
        
        # Create array with exact number of wins and losses
        outcomes = ([True] * n_wins) + ([False] * (n_trials - n_wins))
        
        # Shuffle the outcomes
        np.random.shuffle(outcomes)
        
        return outcomes

    def initialize_block_outcomes(self, n_trials):
        """
        Initialize predetermined outcomes for a block
        """
        half_trials = n_trials // 2
        
        # Generate outcomes for first half
        symbol_a_first = self.generate_fixed_outcomes(half_trials, self.settings['win_probability_good'])
        symbol_b_first = self.generate_fixed_outcomes(half_trials, self.settings['win_probability_bad'])
        
        # Generate outcomes for second half (reversed probabilities)
        symbol_a_second = self.generate_fixed_outcomes(n_trials - half_trials, self.settings['win_probability_bad'])
        symbol_b_second = self.generate_fixed_outcomes(n_trials - half_trials, self.settings['win_probability_good'])
        
        return {
            'first_half': {
                0: symbol_a_first,  # Symbol A outcomes
                1: symbol_b_first   # Symbol B outcomes
            },
            'second_half': {
                0: symbol_a_second,  # Symbol A outcomes
                1: symbol_b_second   # Symbol B outcomes
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
                pos=(0, 0)  # Explicitly set fixation position to center
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
            'timeout_msg': visual.TextStim(
                self.win,
                text='Too Slow, Please Answer Quicker',
                height=0.05,
                wrapWidth=0.8,
                color='red'
            )
        }

    def setup_lsl(self):
        """Setup LSL inlet for R-peak markers"""
        print("Looking for R-peak markers stream...")
        streams = resolve_stream('type', 'R_PEAK')
        self.inlet = StreamInlet(streams[0])

        # Start R-peak collection thread
        self.lsl_thread = threading.Thread(target=self.collect_r_peaks)
        self.lsl_thread.daemon = True
        self.lsl_thread.start()

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
            'n_trials': '60',
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
                keys = event.waitKeys(keyList=['space', 'q'])
                if 'q' in keys:
                    self.win.close()
                    core.quit()
                if 'space' in keys:
                    break

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
        """Generate timing options relative to R-peak with fixed windows"""
        options = {'systole': [], 'diastole': []}
        
        # Create systole grid
        for t in range(self.settings['systole_window_start'], 
                      self.settings['systole_window_end'], 
                      self.settings['grid_resolution']):
            options['systole'].append(r_peak_time + t/1000)
        
        # Create diastole grid
        for t in range(self.settings['diastole_window_start'], 
                      self.settings['diastole_window_end'], 
                      self.settings['grid_resolution']):
            options['diastole'].append(r_peak_time + t/1000)
        
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

    def run_trial(self, trial_num, n_trials, is_reversed, block_outcomes):
        """Run a single trial with conservative cardiac-synced feedback"""
        # Determine timing mode
        timing_mode = self.get_timing_mode(trial_num, n_trials)

        # Send trial start marker
        self.send_marker(self.markers['trial_start'])

        # Decision Phase
        t_0 = time.time()  # Response time

        # Present symbols with fixation cross
        positions = [(-0.15, 0), (0.15, 0)]
        symbol_indices = [0, 1]

        # Randomize positions
        combined = list(zip(self.stim['symbols'], positions, symbol_indices))
        np.random.shuffle(combined)
        symbols, positions, indices = zip(*combined)

        # Draw fixation cross first (it will appear behind the symbols)
        self.stim['fixation'].draw()

        # Then draw symbols
        for sym, pos in zip(symbols, positions):
            sym.pos = pos
            sym.draw()
        self.win.flip()

        # Get response with timeout
        timer = core.Clock()
        keys = event.waitKeys(
            maxWait=self.settings['decision_duration'],
            keyList=['left', 'right', 'q'],
            timeStamped=timer
        )

        if keys and 'q' in keys[0]:
            self.win.close()
            core.quit()

        # Process response
        if not keys:
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
            key, rt = keys[0]
            choice = 0 if key == 'left' else 1
            chosen_symbol = indices[choice]

            # After choice, show only fixation cross for 1-2 seconds
            self.stim['fixation'].draw()
            self.win.flip()
            post_choice_delay = np.random.uniform(1.0, 2.0)
            core.wait(post_choice_delay)

            # Use predetermined outcome based on trial position and chosen symbol
            half = 'second_half' if is_reversed else 'first_half'
            trial_index = trial_num - (n_trials // 2) if is_reversed else trial_num
            outcome = block_outcomes[half][chosen_symbol][trial_index]
            feedback_type = 'win' if outcome else 'loss'

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
            'r_peak_time': r_peak_2,  # Using second R-peak as reference  
            'feedback_time': feedback_time,
            'cardiac_phase': cardiac_phase,
            'time_from_r_peak': feedback_time - r_peak_2,
            'response_time': t_0,
            'first_r_peak': r_peak_1,
            'second_r_peak': r_peak_2
        }

        # Print trial report
        self.print_trial_report(trial_data, trial_num, is_reversed)

        # Inter-trial interval
        self.stim['fixation'].draw()
        self.win.flip()
        core.wait(np.random.uniform(
            self.settings['iti_min'],
            self.settings['iti_max']
        ))

        return trial_data
    
   
    def save_symbol_allocations(self, block_outcomes, participant_info):
        """
        Save symbol allocations for each block to a text file with all trials and correct calculations.
        """
        filename = f"{participant_info['participant']}-ses{participant_info['session']}-run{participant_info['run']}-{participant_info['date_time']}_allocations.txt"
        filepath = self.data_path / filename

        with open(filepath, 'w') as f:
            half_trials = len(block_outcomes['first_half'][0])  # Length of first half
            total_trials = half_trials * 2  # Total number of trials

            for block in range(participant_info['n_blocks']):
                f.write(f"\nBLOCK {block + 1}\n")
                f.write("="*50 + "\n\n")

                # First half
                f.write("First Half (Pre-reversal):\n")
                f.write("-"*30 + "\n")

                first_half_a = block_outcomes['first_half'][0]
                first_half_b = block_outcomes['first_half'][1]

                # Print all trials in first half
                for trial in range(half_trials):
                    f.write(f"Trial {trial:2d}: ")
                    f.write(f"Symbol A -> {'Win ' if first_half_a[trial] else 'Loss'} | ")
                    f.write(f"Symbol B -> {'Win ' if first_half_b[trial] else 'Loss'}\n")

                # Calculate first half statistics
                first_half_a_wins = sum(first_half_a)
                first_half_a_losses = half_trials - first_half_a_wins
                first_half_b_wins = sum(first_half_b)
                first_half_b_losses = half_trials - first_half_b_wins

                f.write("\nFirst Half Summary:\n")
                f.write(f"Symbol A: {first_half_a_wins} Wins, {first_half_a_losses} Losses")
                f.write(f" ({(first_half_a_wins/half_trials)*100:.1f}% Win Rate)\n")
                f.write(f"Symbol B: {first_half_b_wins} Wins, {first_half_b_losses} Losses")
                f.write(f" ({(first_half_b_wins/half_trials)*100:.1f}% Win Rate)\n")

                # Second half
                f.write("\nSecond Half (Post-reversal):\n")
                f.write("-"*30 + "\n")

                second_half_a = block_outcomes['second_half'][0]
                second_half_b = block_outcomes['second_half'][1]

                # Print all trials in second half
                for trial in range(half_trials):
                    f.write(f"Trial {trial + half_trials:2d}: ")
                    f.write(f"Symbol A -> {'Win ' if second_half_a[trial] else 'Loss'} | ")
                    f.write(f"Symbol B -> {'Win ' if second_half_b[trial] else 'Loss'}\n")

                # Calculate second half statistics
                second_half_a_wins = sum(second_half_a)
                second_half_a_losses = half_trials - second_half_a_wins
                second_half_b_wins = sum(second_half_b)
                second_half_b_losses = half_trials - second_half_b_wins

                f.write("\nSecond Half Summary:\n")
                f.write(f"Symbol A: {second_half_a_wins} Wins, {second_half_a_losses} Losses")
                f.write(f" ({(second_half_a_wins/half_trials)*100:.1f}% Win Rate)\n")
                f.write(f"Symbol B: {second_half_b_wins} Wins, {second_half_b_losses} Losses")
                f.write(f" ({(second_half_b_wins/half_trials)*100:.1f}% Win Rate)\n")

                # Block total summary
                f.write("\nBlock Total:\n")
                f.write("-"*30 + "\n")
                f.write(f"Symbol A: {first_half_a_wins + second_half_a_wins} Wins, ")
                f.write(f"{first_half_a_losses + second_half_a_losses} Losses")
                f.write(f" ({((first_half_a_wins + second_half_a_wins)/total_trials)*100:.1f}% Win Rate)\n")
                f.write(f"Symbol B: {first_half_b_wins + second_half_b_wins} Wins, ")
                f.write(f"{first_half_b_losses + second_half_b_losses} Losses")
                f.write(f" ({((first_half_b_wins + second_half_b_wins)/total_trials)*100:.1f}% Win Rate)\n")

                f.write("\n" + "="*50 + "\n")

            print(f"\nSymbol allocations saved to: {filename}") 
    
    
    
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

    def run_experiment(self):
        """Run the complete experiment with markers"""
        # Setup
        participant_info = self.get_participant_info()
        self.setup_experiment()

        # Send experiment start marker
        self.send_marker(self.markers['experiment_start'])

        # Setup LSL
        self.setup_lsl()
        print("Waiting for R-peak stream...")
        core.wait(3.0)

        # Show instructions
        self.show_instructions()

        # Create data storage
        all_data = []

        try:
            # Create a list to store all block outcomes
            all_block_outcomes = []

            # Run blocks
            for block in range(participant_info['n_blocks']):
                # Send block start marker
                self.send_marker(self.markers['block_start'])
                print(f"\nStarting Block {block + 1}...")

                n_trials = participant_info['n_trials']
                reversal_point = n_trials // 2

                # Initialize block outcomes
                block_outcomes = self.initialize_block_outcomes(n_trials)
                # Store block outcomes
                all_block_outcomes.append(block_outcomes)

                for trial in range(n_trials):
                    # Check if we've reached reversal point
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

            # After ALL blocks are completed, save symbol allocations
            for block, block_outcomes in enumerate(all_block_outcomes):
                self.save_symbol_allocations(block_outcomes, participant_info)

        finally:
            # Send experiment end marker
            self.send_marker(self.markers['experiment_end'])
            if all_data:
                df = pd.DataFrame(all_data)
                filename = f"{participant_info['participant']}-ses{participant_info['session']}-run{participant_info['run']}-{participant_info['date_time']}.csv"
                df.to_csv(self.data_path / filename, index=False)
                print(f"\nData saved as: {filename}")
            self.win.close()

if __name__ == "__main__":
    task = CardiacSyncedLearningTask()
    task.run_experiment()