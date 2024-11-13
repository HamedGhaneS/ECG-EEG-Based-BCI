"""
Code to download specific records from the MIT-BIH Arrhythmia Database from PhysioNet

Dataset Description:
The MIT-BIH Arrhythmia Database contains ECG recordings designed to support the development 
and testing of arrhythmia detection algorithms.
- Sampling Rate: 360 Hz
- Typical Length: 30 minutes per record
- Participants: Primarily arrhythmic patients with various types of arrhythmias
- Total Records: 48 carefully selected recordings

This dataset includes records selected to represent both common and uncommon arrhythmias,
making it valuable for training algorithms in arrhythmia detection.

Author: Hamed Ghane
Date: 2024-11-01
"""

import wfdb
import os

# Specify your output directory
# MODIFY THIS PATH according to your system:
# For Windows: Use either of these formats:
#   - "C:\\Users\\YourUsername\\Documents\\MITBIH_Database"
#   - r"C:\Users\YourUsername\Documents\MITBIH_Database"
#   - "C:/Users/YourUsername/Documents/MITBIH_Database"
# For Linux/Mac:
#   - "/home/username/Documents/MITBIH_Database"
#   - "~/Documents/MITBIH_Database"
output_directory = "H:\\Post\\6th_Phase (ECG-EEG Baced BCI)\\2024.11.01\\ECG_Database\\(Arrhytmic+Healthy - 360 Hz) MIT-BIH Arrhythmia Database"

# Create the directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# List of all records in MIT-BIH Arrhythmia Database
records = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234"
]

# Download each record
for record in records:
    try:
        print(f"Downloading record {record}...")
        wfdb.io.dl_database(
            "mitdb", output_directory, records=[record]
        )
        print(f"Successfully downloaded record {record}")
    except Exception as e:
        print(f"Error downloading record {record}: {str(e)}")

print("\nDownload complete. Files saved to:", output_directory)
print(f"Total records attempted: {len(records)}")