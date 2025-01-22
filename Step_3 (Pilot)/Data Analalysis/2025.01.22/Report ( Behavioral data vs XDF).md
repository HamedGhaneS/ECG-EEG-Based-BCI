# Data Coverage Issue and Adjustment Report

## Problem Description
During the analysis of EEG and behavioral data collected from two different systems (PC1 for EEG, PC2 for behavioral), a mismatch in the coverage period was observed. Specifically:
- The behavioral data timestamps were based on **PC2's clock**.
- The EEG data timestamps were based on **PC1's clock**, recorded using LSL LabRecorder.

**Initial Observation (Before Adjustment):**
- **Behavioral Data (PC2):**
  - Start Time: `7003.686 seconds`
  - End Time: `8878.682 seconds`
- **EEG Data (PC1):**
  - Start Time: `6890.767 seconds`
  - End Time: `8839.794 seconds`

This showed that:
- The **EEG recording starts 112.919 seconds earlier** than behavioral data.
- The **EEG recording ends 38.888 seconds earlier** than behavioral data.

## Root Cause
The mismatch arose due to the **time offset between PC1 and PC2 clocks**, causing misaligned timestamps. Based on the provided clock synchronization log (`timing_pairs.csv`), the average offset was calculated as:
- **Mean Offset (PC1 - PC2)**: `-47.22 seconds`
  - This indicates that **PC2 timestamps are ahead of PC1 timestamps by 47.22 seconds**.

## Adjustment Method
To align the EEG data timestamps with the behavioral data timestamps:
- The **mean offset (+47.22 seconds)** was added to all EEG timestamps.

**Adjusted EEG Timings (PC2 clock domain):**
- **EEG Start Time**: `6890.767 + 47.22 = 6937.987 seconds`
- **EEG End Time**: `8839.794 + 47.22 = 8887.014 seconds`

## Coverage After Adjustment
After the adjustment, the alignment was reassessed:
- **Behavioral Data (Fixed on PC2)**:
  - Start Time: `7003.686 seconds`
  - End Time: `8878.682 seconds`
- **Adjusted EEG Data**:
  - Start Time: `6937.987 seconds`
  - End Time: `8887.014 seconds`

**Result**:
- EEG starts **65.699 seconds earlier** than the behavioral data.
- EEG ends **8.332 seconds later** than the behavioral data.

## Conclusion
After applying the adjustment:
- The EEG recording **fully covers the behavioral data collection period**.
- The offset correction aligns the EEG timestamps to the behavioral clock domain, resolving the mismatch.
