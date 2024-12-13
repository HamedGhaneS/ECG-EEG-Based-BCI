# Cardiac-Synced Learning Task Script

## Updates

- **Fixation Cross:**
  - Added randomized ITI fixation cross (1.0–2.0s) at the start of each block and between trials.
  - Ignored responses during fixation periods.

- **Systole-Diastole Timing Grids:**
  - **Systole Timestamps:** 10ms, 150ms, 290ms after R-peak.
  - **Diastole Timestamps:** 310ms, 500ms, 690ms after R-peak.

- **Breaks Between Blocks:**
  - Introduced a 2-minute break between blocks, resumable with a button press.

- **Response Handling:**
  - Cleared response queues before decision phase and ensured responses are logged only during decision phase.

- **Improved Timing Precision:**
  - Optimized CPU usage during fixation and ITI.

This version ensures accurate timing, response handling, and cardiac-synced feedback.
