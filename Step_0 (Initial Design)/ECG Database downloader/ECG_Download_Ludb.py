"""
Code to download the Ludb Single-Lead ECG Database from PhysioNet

Dataset Description:
The Ludb Single-Lead ECG Database contains ECG recordings from both healthy subjects and those with arrhythmias.
- Sampling Rate: 500 Hz
- Typical Length: 5 to 10 minutes per record
- Participants: Includes both healthy individuals and individuals with arrhythmic patterns

This dataset is well-suited for training and testing QRS detection algorithms due to its variety in ECG patterns.

Author: Hamed Ghane
Date: 2024-11-01
"""

import wfdb
import os

# Specify your output directory
# MODIFY THIS PATH according to your system:
# For Windows: Use either of these formats:
#   - "C:\\Users\\YourUsername\\Documents\\Ludb_Database"
#   - r"C:\Users\YourUsername\Documents\Ludb_Database"
#   - "C:/Users/YourUsername/Documents/Ludb_Database"
# For Linux/Mac:
#   - "/home/username/Documents/Ludb_Database"
#   - "~/Documents/Ludb_Database"
output_directory = "H:\\Post\\6th_Phase (ECG-EEG Baced BCI)\\2024.11.01\\Ludb_ECG_Database\\(Healthy+Patient- 500 Hz) Ludb Single-Lead"

# Create the directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Download the entire Ludb Single-Lead ECG Database
wfdb.io.dl_database("ludb", output_directory)

print("Download complete. Files saved to:", output_directory)