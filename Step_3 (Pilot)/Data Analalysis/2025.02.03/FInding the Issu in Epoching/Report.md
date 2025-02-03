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

This resulted in:
1. Timing information being preserved
2. Event IDs being incorrectly reassigned
3. All high_pe trials being grouped at the beginning
4. All low_pe trials being grouped at the end

## Impact
The sorting created an artificial temporal bias where:
- Early trials were predominantly labeled as high_pe
- Late trials were predominantly labeled as low_pe
- This produced a systematic temporal shift in the averaged HEP waveforms

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
3. Observing that the artificial temporal shift in HEP waveforms disappeared

## Conclusion
This case demonstrates how a single line of preprocessing code can introduce systematic biases in EEG/ERP analyses, highlighting the importance of careful data processing validation.
