# Cardiac-Synchronized Learning Task

## Timeline Structure
1. Participant views instructions using yellow button to advance
2. Each participant completes 6 blocks:
   - 40 trials per block
   - 2-minute break between blocks (or until yellow button press)

## Trial Timeline
1. Fixation cross (1.0-2.0s)
2. Choice phase (1.25s):
   - Two symbols shown
   - Left/right response required
3. Post-choice delay (1.5-2.0s)
4. Feedback (4.0s) synchronized to heartbeat:
   - Win = upward arrow
   - Loss = downward arrow

## Block Structure & Outcomes
- Each block uses one timing condition (randomized):
  - Early/Mid/Late Systole (10ms, 150ms, 290ms post R-peak)
  - Early/Mid/Late Diastole (310ms, 500ms, 690ms post R-peak)
- Symbol probabilities:
  - First half: Symbol A (70% win), Symbol B (30% win)
  - Mid-block reversal: Probabilities switch
  - Reversal occurs randomly ±2 trials from block midpoint

## Data Saving
Three files are generated per session:
1. `[participant]-ses[session]-run[run]-[datetime].csv`:
   - Trial-by-trial behavioral data
   - Response times, choices, outcomes
   - Cardiac timing information

2. `[participant]-ses[session]-run[run]-[datetime]_allocations.txt`:
   - Symbol win/loss sequences
   - Pre/post reversal outcomes

3. `[participant]-ses[session]-run[run]-[datetime]_block_order.txt`:
   - Randomized cardiac timing conditions
   - Block-phase mappings
