# Code to download the PTB Diagnostic ECG Database from PhysioNet
# 
# Dataset Description:
# The PTB Diagnostic ECG Database includes ECG recordings from both healthy individuals and patients with various cardiac conditions.
# - Sampling Rate: 1000 Hz
# - Typical Length: 30 seconds to 2 minutes per record
# - Participants: Both healthy controls and patients with cardiac conditions, including arrhythmias, myocardial infarction, etc.
# 
# This dataset is useful for training and testing algorithms across a variety of cardiac conditions due to its high sampling rate and multiple channels.
# 
# Author: Hamed Ghane
# Date: 2024-11-01

import wfdb
import os

# Specify your output directory
output_directory = "H:\\Post\\6th_Phase (ECG-EEG Baced BCI)\\2024.11.01\\ECG Database\\Database\\(Patient+Healthy - 1000 Hz) PTB Diagnostic Database"
os.makedirs(output_directory, exist_ok=True)

# Download the entire PTB Diagnostic ECG Database
wfdb.io.dl_database("ptbdb", output_directory)

print("Download complete. Files saved to:", output_directory)
