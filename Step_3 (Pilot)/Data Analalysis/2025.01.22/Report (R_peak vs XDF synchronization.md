# LSL Clock Synchronization Analysis Report
January 22, 2025

## Setup Overview
- **PC1 (REaltime Analysis and XDF Recording PC)**: Records EEG/ECG data via XDF
- **PC2 (Stimulus PC)**: Runs behavioral task, receives real-time R-peaks

## Timing Analysis
- Mean offset between PC1-PC2 LSL clocks: -47.221s
- Offset standard deviation: 0.005593s
- Stable offset throughout recording (minimal drift)

## Critical Data Sources
1. **XDF File** (PC1 clock domain):
   - Contains EEG/ECG continuous data
   - Timestamps in PC1's LSL clock

2. **CSV Log File** (Both clock domains):
   - Contains R-peak detections with timestamps in both PC1 and PC2 LSL clocks
   - Serves as critical reference for clock synchronization
   - Enables verification of timing accuracy

3. **Behavioral Data** (PC2 clock domain):
   - All task events and markers
   - Timestamps in PC2's LSL clock

## Synchronization Solution
To align EEG/behavioral data, either:
1. Shift EEG timestamps earlier by ~47.22s (PC1 → PC2)
   ```python
   eeg_timestamps_PC2 = eeg_timestamps_PC1 - mean_offset
   ```
2. OR shift behavioral events later by ~47.22s (PC2 → PC1)
   ```python
   event_times_PC1 = event_times_PC2 + mean_offset
   ```

## Validation
- R-peak alignment in visualization confirms timing accuracy
- Small standard deviation confirms stable offset
- Visual inspection shows successful synchronization

## Key Takeaway
The CSV log file recording R-peaks in both clock domains was essential for:
1. Quantifying the exact clock offset
2. Verifying timing stability
3. Enabling precise data synchronization
