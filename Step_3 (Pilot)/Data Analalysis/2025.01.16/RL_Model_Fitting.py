"""
RL Model Fitting Script for Cardiac Learning Study
------------------------------------------------
This script fits a Simple Cue RL model to behavioral data from the cardiac phase learning task.
It processes trial-by-trial choices and outcomes while accounting for cardiac timing information.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path
import json

class SimpleCueRLModel:
    def __init__(self):
        self.param_bounds = [
            (0, 1),     # alpha (learning rate)
            (0, 20),    # beta (inverse temperature)
            (-2, 2)     # gamma (choice stickiness)
        ]
    
    def prepare_data(self, exp_file, timing_file, block_file):
        # Read experimental data
        print(f"Reading experimental data from: {exp_file}")
        behavior_data = pd.read_csv(exp_file)
        
        # Read timing data
        print(f"Reading timing data from: {timing_file}")
        timing_data = pd.read_csv(timing_file)
        
        # Initialize trial sequences
        n_trials = len(behavior_data)
        print(f"Processing {n_trials} trials...")
        
        choices = np.full(n_trials, np.nan)
        outcomes = np.full(n_trials, np.nan)
        valid_trials = np.ones(n_trials, dtype=bool)
        
        # Process each trial
        for trial_idx in range(n_trials):
            trial = behavior_data.iloc[trial_idx]
            
            if trial['feedback'] == 'timeout':
                valid_trials[trial_idx] = False
                print(f"Trial {trial_idx}: Timeout detected")
            else:
                # Convert choices to 0/1
                choices[trial_idx] = 0 if trial['chosen_symbol'] == 'A' else 1
                # Convert outcomes to 0/1
                outcomes[trial_idx] = 1 if trial['feedback'] == 'win' else 0
        
        print(f"Data preparation complete. Found {np.sum(~valid_trials)} timeout trials.")
        
        return {
            'choices': choices,
            'outcomes': outcomes,
            'valid_trials': valid_trials,
            'block': behavior_data['block'].values,
            'cardiac_phase': behavior_data['cardiac_phase'].values,
            'is_reversed': behavior_data['is_reversed'].values
        }

    def run_model(self, trial_data, params):
        print("Running RL model with parameters:", params)
        
        alpha, beta, gamma = params['alpha'], params['beta'], params['gamma']
        n_trials = len(trial_data['choices'])
        n_cues = 2
        
        values = np.zeros((n_trials + 1, n_cues))
        prediction_errors = np.full(n_trials, np.nan)
        abs_prediction_errors = np.full(n_trials, np.nan)
        choice_probs = np.full(n_trials, np.nan)
        
        prev_choice = None
        
        for t in range(n_trials):
            if trial_data['valid_trials'][t]:
                choice = int(trial_data['choices'][t])
                outcome = trial_data['outcomes'][t]
                
                # Compute choice probability
                value_diff = values[t, 0] - values[t, 1]
                stick_term = gamma * (prev_choice == 0) if prev_choice is not None else 0
                choice_prob = 1 / (1 + np.exp(-(beta * value_diff + stick_term)))
                choice_probs[t] = choice_prob if choice == 0 else (1 - choice_prob)
                
                # Compute prediction error
                pe = outcome - values[t, choice]
                prediction_errors[t] = pe
                abs_prediction_errors[t] = np.abs(pe)
                
                values[t + 1] = values[t].copy()
                values[t + 1, choice] += alpha * pe
                
                prev_choice = choice
            else:
                values[t + 1] = values[t].copy()
        
        return {
            'values': values,
            'prediction_errors': prediction_errors,
            'abs_prediction_errors': abs_prediction_errors,
            'choice_probs': choice_probs
        }

    def compute_log_likelihood(self, params_array, trial_data):
        params = {
            'alpha': params_array[0],
            'beta': params_array[1],
            'gamma': params_array[2]
        }
        
        model_results = self.run_model(trial_data, params)
        valid_trials = trial_data['valid_trials']
        log_likelihood = np.sum(np.log(model_results['choice_probs'][valid_trials]))
        
        return -log_likelihood

    def fit_model(self, trial_data):
        print("Starting model fitting...")
        
        initial_params = [0.3, 1.0, 0.1]  # alpha, beta, gamma
        
        result = minimize(
            self.compute_log_likelihood,
            initial_params,
            args=(trial_data,),
            bounds=self.param_bounds,
            method='L-BFGS-B'
        )
        
        best_params = {
            'alpha': result.x[0],
            'beta': result.x[1],
            'gamma': result.x[2]
        }
        
        print("Model fitting complete. Best parameters found:", best_params)
        
        return best_params, self.run_model(trial_data, best_params)

def classify_trials_for_hep(model_results, trial_data):
    print("Classifying trials for HEP analysis...")
    
    pe = model_results['prediction_errors']
    abs_pe = model_results['abs_prediction_errors']
    valid_trials = trial_data['valid_trials']
    
    valid_pe = pe[valid_trials]
    valid_abs_pe = abs_pe[valid_trials]
    median_abs_pe = np.median(valid_abs_pe)
    
    classifications = {
        'positive_pe': pe > 0,
        'negative_pe': pe < 0,
        'high_abs_pe': abs_pe > median_abs_pe,
        'low_abs_pe': abs_pe < median_abs_pe,
        'valid_trials': valid_trials
    }
    
    print("Trial classification complete.")
    return classifications


def get_file_metadata(exp_file):
    """
    Extract useful metadata from experiment filename.
    This helps create organized and informative result files.
    """
    # Parse the experiment filename to get participant ID and timestamp
    file_stem = exp_file.stem
    parts = file_stem.split('-')
    participant_id = parts[0]
    timestamp = parts[-1]
    
    return {
        'participant_id': participant_id,
        'timestamp': timestamp
    }

def main():
    print("Starting RL model fitting analysis...")
    
    # Define paths - keep your existing data paths
    base_path = Path(r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.14\Pilot\Harry")
    exp_file = base_path / "Harry-ses001-run1-20250114-130445.csv"
    timing_file = base_path / "Harry-ses001-run1-20250114-130445_timing_pairs.csv"
    block_file = base_path / "Harry-ses001-run1-20250114-130445_block_order.txt"
    
    # Validate file existence
    for file_path in [exp_file, timing_file, block_file]:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Initialize model and process data
    model = SimpleCueRLModel()
    trial_data = model.prepare_data(exp_file, timing_file, block_file)
    
    # Fit model
    best_params, model_results = model.fit_model(trial_data)
    
    # Get script directory for saving results
    script_dir = Path(__file__).parent
    print(f"\nScript is located at: {script_dir}")
    
    # Create results directory in script location
    output_dir = script_dir / "rl_model_results"
    output_dir.mkdir(exist_ok=True)
    print(f"Creating results directory at: {output_dir}")
    
    # Get metadata for filename
    metadata = get_file_metadata(exp_file)
    output_file = output_dir / f"{metadata['participant_id']}_rl_model_results_{metadata['timestamp']}.json"
    
    # Prepare results dictionary
    results = {
        'metadata': {
            'participant_id': metadata['participant_id'],
            'analysis_timestamp': pd.Timestamp.now().strftime('%Y%m%d-%H%M%S'),
            'data_source': str(exp_file),
            'model_type': 'Simple Cue RL'
        },
        'parameters': best_params,
        'prediction_errors': model_results['prediction_errors'].tolist(),
        'abs_prediction_errors': model_results['abs_prediction_errors'].tolist(),
        'trial_classifications': {
            k: v.tolist() for k, v in classify_trials_for_hep(model_results, trial_data).items()
        }
    }
    
    # Save results
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nAnalysis Results Summary:")
    print("-" * 50)
    print("Best fitting parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value:.3f}")
    
    print(f"\nFull results saved to: {output_file}")

if __name__ == "__main__":
    main()
