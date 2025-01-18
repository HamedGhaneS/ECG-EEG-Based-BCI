# HEP-RLParameter Analysis Report

## 1. What I Was Trying to Do:
- Analyze HEP (Heartbeat Evoked Potentials) from EEG data.
- Use R-peak timings from real-time detection (stored in `timing_pairs.csv`).
- Compare different conditions:
  - Positive vs negative PE.
  - High vs low PE.
  - Correct vs incorrect trials.
- Focus analysis on the first three R-peaks after feedback onset.

## 2. The Issue I Encountered:
- The script successfully loads the data but produces **0 epochs** for all conditions.
- This suggests a **timing alignment issue** between:
  - The XDF file (EEG data, `2283.860` seconds long).
  - The behavioral data (trial timings).
  - The R-peak timings from PC2.

## 3. The Next Step I Should Try:
- Adding the RL model parameters to the trial segmentation file; Including:
    - AbsPE
    - SignedPE
    - Correct vs Incorrect
- Investigate the **timing alignment** between:
  - XDF file recording.
  - Trial behavioral data.
  - R-peak timings.
- This investigation will help ensure proper alignment and extraction of epochs.

## 4. Key Files I Am Working With:
- **XDF file**: `sub-P001_ses-S001_task-Default_run-001_eeg.xdf`
- **Behavioral data**: `Harry-ses001-run1-20250114-130445.csv`
- **R-peak timings**: `trial_timing_analysis_20250117_150100.csv`
- **RL model results**: `Harry_rl_model_results_130445.json`

## Notes:
The last script I used was for manual epoch extraction.  
Before using it, I need to verify the **timing alignment** between all these data sources.  

**Plan for Tomorrow:**  
Start investigating the timing alignment.
