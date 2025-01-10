## Cardiac Synced Learning Task Script - Output Log Report

### 1. General Information
The script implements a cardiac-synchronized learning task using **PsychoPy**, **LSL (Lab Streaming Layer)**, and **Cedrus XID** devices to deliver stimuli and collect responses. It includes randomized block outcomes, timing synchronization based on R-peak detection, and generates detailed logs for each trial.

### 2. Output File Logs
**Log Types:**
1. **Data File (CSV):** Contains trial-wise data, including reaction times, choice outcomes, cardiac timing information, and timestamps.
2. **Timing Pairs (CSV):** Logs of R-peak timing pairs collected during the experiment, including timing in both PC LSL local clock to measure the LSL clock offset, for post experiment synchronization analysis.
3. **Block Order File (TXT):** Detailed block order information, including randomized conditions and cardiac phases.
4. **Symbol Allocations (TXT):** Summary of symbol allocations and outcome probabilities before and after reversal points.

### 3. Data File (CSV)
**Columns:**
- `trial_start_time`: Start time of the trial in LSL timebase.
- `symbols_onset_time`: Timestamp when the symbols were displayed.
- `choice_time`: Time when the participant made a choice.
- `rt`: Reaction time in seconds.
- `choice`: Index of the chosen symbol (0 or 1).
- `chosen_symbol`: 'A' or 'B' indicating the selected symbol.
- `feedback`: Type of feedback provided ('win', 'loss', or 'neutral').
- `block`: Block number (starting from 0).
- `trial`: Trial number within the block.
- `reversal_point`: Trial number where probability reversal occurred.
- `block_condition`: Timing condition (e.g., 'systole-mid').
- `percentage_rr`: R-R interval percentage for timing calculation.
- `intended_presentation_time`: Planned time for feedback presentation.
- `actual_presentation_time`: Actual time when feedback was presented.
- `timing_precision_ms`: Difference between intended and actual presentation times in milliseconds.

### 4. Timing Pairs File (CSV)
**Columns:**
- `peak_number`: Sequential number of detected R-peaks.
- `pc1_lsl_time`: R-peak timestamp from LSL.
- `pc2_lsl_time`: Corresponding local clock time.
- `calculated_offset`: Time difference between LSL and local clock.
- `rr_interval`: Calculated R-R interval.
- `heart_rate`: Instantaneous heart rate (BPM).
- `time_recorded`: Local time when the R-peak was recorded.

### 5. Block Order File (TXT)
**Contents:**
- Block-wise randomized order of conditions.
- Cardiac phase and timing percentages.
- R-R percentage for feedback presentation.

### 6. Symbol Allocations File (TXT)
**Contents:**
- Outcome probabilities for each block, both before and after the reversal point.
- Summary of win/loss outcomes for symbols A and B.
- Reversal point for each block.

### 7. Example Log Snippets
**Sample Trial Data (CSV):**
| trial_start_time | symbols_onset_time | choice_time | rt  | choice | chosen_symbol | feedback | block | trial | reversal_point | block_condition | percentage_rr | intended_presentation_time | actual_presentation_time | timing_precision_ms |
|------------------|--------------------|-------------|-----|--------|---------------|----------|-------|-------|----------------|-----------------|---------------|---------------------------|--------------------------|---------------------|
| 12.345           | 12.678             | 13.456      | 0.778 | 1      | B             | win      | 0     | 1     | 5              | systole-mid     | 0.21          | 13.567                     | 13.570                   | 3.0                 |

**Sample Timing Pair Data (CSV):**
| peak_number | pc1_lsl_time | pc2_lsl_time | calculated_offset | rr_interval | heart_rate | time_recorded |
|-------------|--------------|--------------|-------------------|-------------|------------|---------------|
| 1           | 12.345       | 12.350       | -0.005            | 0.850       | 70.6       | 14:32:12      |

**Sample Block Order File (TXT):**
```
Block Order Information
=====================

Block 1:
  Original Condition: Block 1
  Cardiac Phase: systole
  Timing Point: early
  R-R Percentage: 6.0%
-----------------------------

Block 2:
  Original Condition: Block 2
  Cardiac Phase: diastole
  Timing Point: mid
  R-R Percentage: 70.0%
-----------------------------
```

### 8. Error Handling and Warnings
- **Missing R-Peaks:** If no valid R-peaks are detected within the timeout period, immediate feedback is displayed without cardiac synchronization.
- **LSL Stream Errors:** If the LSL stream fails to initialize or disconnects during the experiment, an error message is logged, and the experiment terminates gracefully.
- **Data Saving Issues:** Any errors during data saving are logged with a warning message specifying the type of error.

### 9. Cleanup Sequence
At the end of the experiment or in case of an error, the following cleanup actions are performed:
1. **Stopping R-peak collection thread.**
2. **Closing LSL inlet stream.**
3. **Saving all collected data.**
4. **Closing PsychoPy window.**
5. **Resetting Cedrus device.**

### 10. Conclusion
This report outlines the structure of output logs generated by the script, including data files, timing pairs, block order, and symbol allocations. Detailed information ensures reproducibility and accurate post-experiment analysis.

---

