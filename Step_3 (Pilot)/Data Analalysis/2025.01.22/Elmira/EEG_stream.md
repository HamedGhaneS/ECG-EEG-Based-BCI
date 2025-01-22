# EEG Stream Specifications in XDF Recording

## Stream Information
- **Stream Number:** **4** (BrainAmpSeries)
- **Stream Type:** EEG
- **Stream ID:** 4

## Technical Parameters
- **Sampling Rate:** 5000 Hz
- **Number of Channels:** 64
- **Total Samples:** 9,745,250

## Recording Duration
- **Start Time:** 6392.45s
- **Data Shape:** (9745250, 64)

## Stream Structure
The EEG stream is stored as a continuous time series with:
- Data stored in numpy.ndarray format
- Each sample containing 64 channel values
- Timestamps aligned with each sample point
- Continuous recording without gaps

## Note
This is the main EEG data stream in the XDF file, distinguished from other streams which include:
1. R-peak markers
2. Task markers
3. BrainAmp markers
