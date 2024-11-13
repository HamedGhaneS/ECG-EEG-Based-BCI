# Real-Time ECG-EEG BCI for Cardiac Cycle Modulation in Outcome Presentation

This project is a real-time ECG-EEG-based Brain-Computer Interface (BCI) experiment aimed at investigating the influence of cardiac phase timing on brain activity, specifically through Heartbeat-Evoked Potentials (HEP), in the context of outcome presentation. The experiment is based on the findings of the paper **[Timing along the cardiac cycle modulates neural signals of reward-based learning](https://www.nature.com/articles/s41467-024-46921-5)** by Fouragnan et al. (2024), which highlights the distinct neural responses associated with systole and diastole phases during learning processes.

## Background

The referenced study by Fouragnan et al. (2024) suggests that the timing of reward or outcome presentation along the cardiac cycle, particularly during systole and diastole phases, affects neural responses related to reward processing. This project aims to validate and extend these findings by focusing specifically on the diastole phase to observe potential differences in brain responses. Using a real-time BCI setup, we present outcomes during different phases of the cardiac cycle to examine variations in HEP during the diastole phase.

## Objective

The primary objective of this project is to determine if there is a significant effect on HEP **at different points within the diastole phase** in response to reward-based outcomes. This real-time analysis extends beyond comparing systole and diastole effects, aiming to explore potential variations in reward-related brain activity specifically within the diastole phase, depending on the exact timing of outcome presentation.

## Features

- **Real-Time Cardiac Phase Detection**: Continuous ECG monitoring to identify R-peaks in real-time, marking the systole and diastole phases of the cardiac cycle.
- **Cardiac Phase Synchronization**: Synchronizes EEG data with detected ECG R-peaks to enable precise phase-based outcome presentation.
- **Dynamic Outcome Presentation**: Adjusts the timing of outcome presentation based on the participant’s cardiac cycle, particularly targeting diastole.
- **Real-Time EEG Processing with LSL**: Utilizes the Lab Streaming Layer (LSL) for real-time streaming of EEG and ECG data, enabling minimal delay in phase-based stimulus delivery and response monitoring.
- **Heartbeat-Evoked Potential (HEP) Analysis**: Analyzes EEG responses within the diastole phase to identify any HEP-related changes due to timing within the cardiac cycle.

## Setup and Configuration

1. **Hardware**: 
   - BrainAmp amplifier and BrainCap MR system (64 EEG electrodes + 1 ECG electrode).
   - Real-time ECG and EEG data acquisition for precise timing within the cardiac cycle.
   
2. **Software**:
   - Lab Streaming Layer (LSL) for real-time data streaming and synchronization of EEG and ECG signals.
   - Custom scripts to process ECG and EEG signals in real-time and identify cardiac phases.
   - Real-time data analysis tools to process HEP responses during the diastole phase.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/HamedGhaneS/ECG-EEG-Based-BCI.git
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Associated Laboratory

This research is conducted in collaboration with [PhiliasTidesLab](https://github.com/PhiliasTidesLab).

## Reference

The project builds on the work of Fouragnan et al. (2024), published in **Nature Communications**: [Timing along the cardiac cycle modulates neural signals of reward-based learning](https://www.nature.com/articles/s41467-024-46921-5). This study offers insights into the relationship between cardiac phase timing and reward-based learning, particularly the differential neural responses in systole and diastole.

