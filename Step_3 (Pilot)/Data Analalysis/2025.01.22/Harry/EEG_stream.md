# EEG Stream Specifications in XDF Recording

## Stream Information
- **Stream Number:** 3 (BrainAmpSeries) 
- **Stream Type:** EEG
- **Stream ID:** 3

## Technical Parameters
- **Sampling Rate:** 5000 Hz
- **Number of Channels:** 64  
- **Total Samples:** 11,419,300

## Recording Duration
- **Start Time:** 4353.00s
- **Data Shape:** (11419300, 64)

## Stream Structure
The EEG stream is stored as a continuous time series with:
- Data stored in numpy.ndarray format
- Each sample containing 64 channel values 
- Timestamps aligned with each sample point
- Continuous recording without gaps

## Note
This is the main EEG data stream in the XDF file, distinguished from other streams which include:
1. TaskMarkers (empty)
2. BrainAmp markers (empty)
