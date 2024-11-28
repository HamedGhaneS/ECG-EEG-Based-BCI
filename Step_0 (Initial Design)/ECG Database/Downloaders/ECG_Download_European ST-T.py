"""
Code to download the European ST-T Database from PhysioNet

Dataset Description:
The European ST-T Database includes annotated ECG recordings primarily from patients, focusing on ST and T-wave changes.
- Sampling Rate: 250 Hz
- Typical Length: 2 hours per record
- Participants: Primarily patients with abnormal ECG patterns, including those with various types of ST and T-wave changes.

This dataset is valuable for studying ECG changes related to myocardial ischemia and other conditions affecting the ST-T segment.

Author: Hamed Ghane
Date: 2024-11-01
"""

import wfdb
import os

# Specify your output directory
# MODIFY THIS PATH according to your system:
# For Windows: Use either of these formats:
#   - "C:\\Users\\YourUsername\\Documents\\ECG_Database"
#   - r"C:\Users\YourUsername\Documents\ECG_Database"
#   - "C:/Users/YourUsername/Documents/ECG_Database"
output_directory = "H:\\Post\\6th_Phase (ECG-EEG Baced BCI)\\2024.11.01\\ECG Database\\Database\\(Patients +Healthy - 250 Hz) European ST-T Database"

# Create the directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Download the entire European ST-T Database
wfdb.io.dl_database("edb", output_directory)

print("Download complete. Files saved to:", output_directory)
