Here's the updated report with figure placeholders:

# Analysis Issue Report: Event Sorting Effect on EEG Epoching

## Issue Identification
During Heart-Evoked Potential (HEP) analysis, an unexpected temporal shift was observed between conditions (high vs. low PE) across all participants. This shift manifested as a consistent temporal delay between conditions, indicating a potential systematic bias in the epoching process.

## Root Cause Analysis
The issue was traced to a single line in the epoching script:
```python
events = np.sort(events, axis=0)
```

This command incorrectly sorted each column of the events array independently:
```python
# Original events array:
events = [
    [100, 0, 2],  # sample=100, trigger=0, event_id=2 (low_pe)
    [200, 0, 1],  # sample=200, trigger=0, event_id=1 (high_pe)
    [300, 0, 2],  # sample=300, trigger=0, event_id=2 (low_pe)
]

# After np.sort(events, axis=0):
events = [
    [100, 0, 1],  # sample times preserved but event_ids reordered
    [200, 0, 1],
    [300, 0, 2]
]
```

## Impact & Visualization
The sorting created an artificial temporal bias demonstrated in the figures below:

![Figure 1: HEP analysis with incorrect sorting](Step_3 (Pilot)/Data Analalysis/2025.02.03/FInding the Issu in Epoching/Before.png)
*Figure 1: HEP analysis with event sorting. Note the systematic temporal shift between high_pe (red) and low_pe (blue) conditions, particularly around the R-peak (t=0).*

![Figure 2: HEP analysis after removing sorting](Step_3 (Pilot)/Data Analalysis/2025.02.03/FInding the Issu in Epoching/After.png)
*Figure 2: HEP analysis without event sorting. The temporal alignment between conditions is now correct, showing proper synchronization around the R-peak.*

The impact of sorting resulted in:
- Early trials being predominantly labeled as high_pe
- Late trials being predominantly labeled as low_pe
- Creating a systematic temporal shift in the averaged HEP waveforms

## Solution
Removing the sorting line preserved the natural experimental sequence:
```python
# Convert events to numpy array WITHOUT sorting
events = np.array(events, dtype=int)
```

## Verification
The solution was verified by:
1. Comparing epoch timing information with and without sorting
2. Confirming proper condition labeling matches the original behavioral data
3. Observing that the artificial temporal shift in HEP waveforms disappeared (see Figure 2)

## Conclusion
This case demonstrates how a single line of preprocessing code can introduce systematic biases in EEG/ERP analyses, highlighting the importance of careful data processing validation. The corrected analysis reveals the true temporal relationships between conditions in the HEP data.
