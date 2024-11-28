"""
Code to download the MIT-BIH Normal Sinus Rhythm Database from PhysioNet

Dataset Description:
The MIT-BIH Normal Sinus Rhythm Database contains ECG recordings from subjects 
who had no significant arrhythmias.
- Sampling Rate: 128 Hz
- Population: Healthy individuals with normal sinus rhythm
- Total Records: 18 long-term ECG recordings
- Purpose: Provides baseline data for normal heart rhythm studies

This dataset is valuable for:
- Establishing normal ECG baselines
- Comparing against arrhythmic patterns
- Training algorithms to recognize normal sinus rhythm

Author: Hamed Ghane
Date: 2024-11-01
"""

import wfdb
import os

# Specify your output directory
# MODIFY THIS PATH according to your system:
# For Windows: Use either of these formats:
#   - "C:\\Users\\YourUsername\\Documents\\NSRDB"
#   - r"C:\Users\YourUsername\Documents\NSRDB"
#   - "C:/Users/YourUsername/Documents/NSRDB"
# For Linux/Mac:
#   - "/home/username/Documents/NSRDB"
#   - "~/Documents/NSRDB"
output_directory = "H:\\Post\\6th_Phase (ECG-EEG Baced BCI)\\2024.11.01\\ECG_Database"

# Create the directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# List of records in MIT-BIH Normal Sinus Rhythm Database
healthy_records = [
    "16265", "16272", "16273", "16420", "16483", "16539", "16773", "16786",
    "16795", "17052", "17453", "18177", "18184", "19088", "19090", "19093",
    "19140", "19830"
]

# Download each healthy record with error handling
successful_downloads = 0
failed_downloads = 0

for record in healthy_records:
    try:
        print(f"Downloading record {record}...")
        wfdb.io.dl_database(
            "nsrdb", output_directory, records=[record]
        )
        successful_downloads += 1
        print(f"Successfully downloaded record {record}")
    except Exception as e:
        failed_downloads += 1
        print(f"Error downloading record {record}: {str(e)}")

# Print summary
print("\nDownload Summary:")
print(f"Total records attempted: {len(healthy_records)}")
print(f"Successfully downloaded: {successful_downloads}")
print(f"Failed downloads: {failed_downloads}")
print(f"\nFiles saved to: {output_directory}")
