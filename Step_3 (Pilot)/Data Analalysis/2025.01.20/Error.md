# The last terminal log

--------------------

Loading XDF file...
Creating MNE Raw object...
Creating RawArray with float64 data, n_channels=64, n_times=11419300
    Range : 0 ... 11419299 =      0.000 ...  2283.860 secs
Ready.
Loading timing data...
Preprocessing data...
Filtering raw data in 1 contiguous segment
Setting up band-pass filter from 0.1 - 40 Hz

FIR filter parameters
---------------------
Designing a one-pass, zero-phase, non-causal bandpass filter:
- Windowed time-domain design (firwin) method
- Hamming window with 0.0194 passband ripple and 53 dB stopband attenuation
- Lower passband edge: 0.10
- Lower transition bandwidth: 0.10 Hz (-6 dB cutoff frequency: 0.05 Hz)
- Upper passband edge: 40.00 Hz
- Upper transition bandwidth: 10.00 Hz (-6 dB cutoff frequency: 45.00 Hz)
- Filter length: 165001 samples (33.000 s)

[Parallel(n_jobs=1)]: Done  17 tasks      | elapsed:   14.6s
Filtering raw data in 1 contiguous segment
Setting up band-stop filter

FIR filter parameters
---------------------
Designing a one-pass, zero-phase, non-causal bandstop filter:
- Windowed time-domain design (firwin) method
- Hamming window with 0.0194 passband ripple and 53 dB stopband attenuation
- Lower transition bandwidth: 0.50 Hz
- Upper transition bandwidth: 0.50 Hz
- Filter length: 33001 samples (6.600 s)

[Parallel(n_jobs=1)]: Done  17 tasks      | elapsed:    6.3s
EEG channel type selected for re-referencing
Applying average reference.
Applying a custom ('EEG',) reference.
Cleaning data with ICA...
Fitting ICA to data using 63 channels (please be patient, this may take a while)
Selecting by number: 20 components
Fitting ICA took 261.6s.
Using threshold: 0.08 for CTPS ECG detection
Using channel ECG to identify heart beats.
Setting up band-pass filter from 8 - 16 Hz

FIR filter parameters
---------------------
Designing a two-pass forward and reverse, zero-phase, non-causal bandpass filter:
- Windowed frequency-domain design (firwin2) method
- Hann window
- Lower passband edge: 8.00
- Lower transition bandwidth: 0.50 Hz (-12 dB cutoff frequency: 7.75 Hz)
- Upper passband edge: 16.00 Hz
- Upper transition bandwidth: 0.50 Hz (-12 dB cutoff frequency: 16.25 Hz)
- Filter length: 50000 samples (10.000 s)

Number of ECG events detected : 2619 (average pulse 68 / min.)
Not setting metadata
2619 matching events found
No baseline correction applied
Using data from preloaded Raw for 2619 events and 5001 original time points ...
1 bad epochs dropped
Found 2 ECG components
Applying ICA to Raw instance
    Transforming to ICA space (20 components)
    Zeroing out 2 ICA components
    Projecting back using 63 PCA components
Creating events...
All events created:
[[ 165    0    2]
 [ 190    0    2]
 [ 200    0    2]
 [ 214    0    1]
 [ 220    0    1]
 [ 235    0    2]
 [ 300    0    2]
 [ 300    0    2]
 [ 300    0    2]
 [ 310    0    1]
 [ 315    0    2]
 [ 345    0    1]
 [ 375    0    1]
 [ 380    0    2]
 [ 390    0    2]
 [ 400    0    2]
 [ 415    0    1]
 [ 425    0    2]
 [ 425    0    2]
 [ 434    0    2]
 [ 440    0    1]
 [ 445    0    1]
 [ 450    0    1]
 [ 450    0    2]
 [ 475    0    1]
 [ 485    0    1]
 [ 485    0    1]
 [ 525    0    2]
 [ 530    0    2]
 [ 535    0    1]
 [ 565    0    1]
 [ 590    0    1]
 [ 595    0    1]
 [ 640    0    1]
 [ 640    0    1]
 [ 740    0    2]
 [ 750    0    1]
 [ 855    0    2]
 [ 900    0    2]
 [ 910    0    2]
 [ 945    0    2]
 [ 950    0    2]
 [1065    0    2]
 [1090    0    2]
 [1095    0    1]
 [1095    0    1]
 [1100    0    2]
 [1100    0    2]
 [1125    0    1]
 [1125    0    2]
 [1135    0    2]
 [1150    0    1]
 [1175    0    1]
 [1195    0    1]
 [1215    0    2]
 [1245    0    1]
 [1260    0    2]
 [1300    0    2]
 [1345    0    1]
 [1345    0    2]
 [1355    0    1]
 [1355    0    2]
 [1370    0    1]
 [1390    0    1]
 [1395    0    2]
 [1424    0    1]
 [1429    0    2]
 [1445    0    1]
 [1445    0    2]
 [1450    0    1]
 [1450    0    2]
 [1465    0    2]
 [1475    0    1]
 [1480    0    2]
 [1490    0    2]
 [1500    0    1]
 [1595    0    2]
 [1610    0    1]
 [1830    0    1]
 [1875    0    1]
 [1895    0    1]
 [1900    0    1]
 [1975    0    2]
 [2000    0    2]
 [2005    0    1]
 [2055    0    2]
 [2060    0    1]
 [2090    0    2]
 [2095    0    1]
 [2100    0    2]
 [2110    0    1]
 [2110    0    2]
 [2120    0    1]
 [2120    0    2]
 [2150    0    1]
 [2160    0    1]
 [2170    0    1]
 [2175    0    2]
 [2190    0    2]
 [2190    0    2]
 [2195    0    1]
 [2205    0    1]
 [2210    0    1]
 [2225    0    2]
 [2230    0    2]
 [2235    0    1]
 [2260    0    2]
 [2275    0    1]
 [2275    0    2]
 [2275    0    2]
 [2290    0    1]
 [2290    0    1]
 [2290    0    2]
 [2300    0    2]
 [2310    0    1]
 [2310    0    1]
 [2340    0    1]
 [2340    0    2]
 [2380    0    2]
 [2385    0    1]
 [2425    0    1]
 [2495    0    2]
 [2535    0    2]
 [2555    0    1]
 [2580    0    2]
 [2585    0    1]
 [2635    0    1]
 [2655    0    1]
 [2670    0    2]
 [2670    0    2]
 [2695    0    2]
 [2720    0    1]
 [2740    0    2]
 [2745    0    1]
 [2745    0    2]
 [2750    0    2]
 [2755    0    1]
 [2800    0    2]
 [2805    0    1]
 [2810    0    1]
 [2810    0    2]
 [2824    0    1]
 [2824    0    2]
 [2834    0    2]
 [2839    0    1]
 [2839    0    1]
 [2839    0    1]
 [2849    0    1]
 [2875    0    1]
 [2885    0    1]
 [2900    0    2]
 [2915    0    1]
 [2990    0    2]
 [3160    0    2]
 [3160    0    2]
 [3180    0    1]
 [3210    0    1]
 [3230    0    1]
 [3250    0    1]
 [3260    0    1]
 [3260    0    2]
 [3270    0    2]
 [3285    0    1]
 [3300    0    1]
 [3305    0    1]
 [3310    0    1]
 [3310    0    2]
 [3310    0    2]
 [3360    0    1]
 [3385    0    2]
 [3390    0    2]
 [3390    0    2]
 [3400    0    2]
 [3435    0    2]
 [3439    0    1]
 [3439    0    1]
 [3439    0    1]
 [3459    0    2]
 [3469    0    2]
 [3484    0    2]
 [3489    0    1]
 [3500    0    2]
 [3500    0    2]
 [3525    0    2]
 [3545    0    1]
 [3545    0    2]
 [3555    0    1]
 [3555    0    2]
 [3565    0    2]
 [3570    0    1]
 [3580    0    2]
 [3600    0    2]
 [3605    0    1]
 [3615    0    1]
 [3665    0    2]
 [3705    0    2]
 [3780    0    1]
 [3785    0    2]
 [3825    0    1]
 [3870    0    2]
 [3895    0    1]
 [3920    0    1]
 [3950    0    1]
 [3950    0    2]
 [3955    0    1]
 [3955    0    2]
 [3955    0    2]
 [3960    0    1]
 [3965    0    1]
 [3985    0    2]
 [3995    0    2]
 [4015    0    2]
 [4040    0    1]
 [4045    0    2]
 [4060    0    1]
 [4094    0    1]
 [4094    0    2]
 [4094    0    2]
 [4100    0    1]
 [4105    0    2]
 [4110    0    1]
 [4110    0    2]
 [4110    0    2]
 [4110    0    2]
 [4120    0    1]
 [4160    0    1]
 [4170    0    2]
 [4185    0    1]
 [4190    0    1]
 [4190    0    2]
 [4195    0    1]
 [4310    0    1]
 [4335    0    1]
 [4335    0    1]
 [4360    0    2]
 [4370    0    2]
 [4520    0    1]]

Duplicate events detected:
Sample: 300, Event ID: 2
Sample: 300, Event ID: 2
Sample: 425, Event ID: 2
Sample: 450, Event ID: 2
Sample: 485, Event ID: 1
Sample: 640, Event ID: 1
Sample: 1095, Event ID: 1
Sample: 1100, Event ID: 2
Sample: 1125, Event ID: 2
Sample: 1345, Event ID: 2
Sample: 1355, Event ID: 2
Sample: 1445, Event ID: 2
Sample: 1450, Event ID: 2
Sample: 2110, Event ID: 2
Sample: 2120, Event ID: 2
Sample: 2190, Event ID: 2
Sample: 2275, Event ID: 2
Sample: 2275, Event ID: 2
Sample: 2290, Event ID: 1
Sample: 2290, Event ID: 2
Sample: 2310, Event ID: 1
Sample: 2340, Event ID: 2
Sample: 2670, Event ID: 2
Sample: 2745, Event ID: 2
Sample: 2810, Event ID: 2
Sample: 2824, Event ID: 2
Sample: 2839, Event ID: 1
Sample: 2839, Event ID: 1
Sample: 3160, Event ID: 2
Sample: 3260, Event ID: 2
Sample: 3310, Event ID: 2
Sample: 3310, Event ID: 2
Sample: 3390, Event ID: 2
Sample: 3439, Event ID: 1
Sample: 3439, Event ID: 1
Sample: 3500, Event ID: 2
Sample: 3545, Event ID: 2
Sample: 3555, Event ID: 2
Sample: 3950, Event ID: 2
Sample: 3955, Event ID: 2
Sample: 3955, Event ID: 2
Sample: 4094, Event ID: 2
Sample: 4094, Event ID: 2
Sample: 4110, Event ID: 2
Sample: 4110, Event ID: 2
Sample: 4110, Event ID: 2
Sample: 4190, Event ID: 2
Sample: 4335, Event ID: 1
Analyzing HEP...
Creating epochs from 237 events...
Error while creating epochs: Event time samples were not unique. Consider setting the `event_repeated` parameter."
Events causing the error:
[[ 165    0    2]
 [ 190    0    2]
 [ 200    0    2]
 [ 214    0    1]
 [ 220    0    1]
 [ 235    0    2]
 [ 300    0    2]
 [ 300    0    2]
 [ 300    0    2]
 [ 310    0    1]
 [ 315    0    2]
 [ 345    0    1]
 [ 375    0    1]
 [ 380    0    2]
 [ 390    0    2]
 [ 400    0    2]
 [ 415    0    1]
 [ 425    0    2]
 [ 425    0    2]
 [ 434    0    2]
 [ 440    0    1]
 [ 445    0    1]
 [ 450    0    1]
 [ 450    0    2]
 [ 475    0    1]
 [ 485    0    1]
 [ 485    0    1]
 [ 525    0    2]
 [ 530    0    2]
 [ 535    0    1]
 [ 565    0    1]
 [ 590    0    1]
 [ 595    0    1]
 [ 640    0    1]
 [ 640    0    1]
 [ 740    0    2]
 [ 750    0    1]
 [ 855    0    2]
 [ 900    0    2]
 [ 910    0    2]
 [ 945    0    2]
 [ 950    0    2]
 [1065    0    2]
 [1090    0    2]
 [1095    0    1]
 [1095    0    1]
 [1100    0    2]
 [1100    0    2]
 [1125    0    1]
 [1125    0    2]
 [1135    0    2]
 [1150    0    1]
 [1175    0    1]
 [1195    0    1]
 [1215    0    2]
 [1245    0    1]
 [1260    0    2]
 [1300    0    2]
 [1345    0    1]
 [1345    0    2]
 [1355    0    1]
 [1355    0    2]
 [1370    0    1]
 [1390    0    1]
 [1395    0    2]
 [1424    0    1]
 [1429    0    2]
 [1445    0    1]
 [1445    0    2]
 [1450    0    1]
 [1450    0    2]
 [1465    0    2]
 [1475    0    1]
 [1480    0    2]
 [1490    0    2]
 [1500    0    1]
 [1595    0    2]
 [1610    0    1]
 [1830    0    1]
 [1875    0    1]
 [1895    0    1]
 [1900    0    1]
 [1975    0    2]
 [2000    0    2]
 [2005    0    1]
 [2055    0    2]
 [2060    0    1]
 [2090    0    2]
 [2095    0    1]
 [2100    0    2]
 [2110    0    1]
 [2110    0    2]
 [2120    0    1]
 [2120    0    2]
 [2150    0    1]
 [2160    0    1]
 [2170    0    1]
 [2175    0    2]
 [2190    0    2]
 [2190    0    2]
 [2195    0    1]
 [2205    0    1]
 [2210    0    1]
 [2225    0    2]
 [2230    0    2]
 [2235    0    1]
 [2260    0    2]
 [2275    0    1]
 [2275    0    2]
 [2275    0    2]
 [2290    0    1]
 [2290    0    1]
 [2290    0    2]
 [2300    0    2]
 [2310    0    1]
 [2310    0    1]
 [2340    0    1]
 [2340    0    2]
 [2380    0    2]
 [2385    0    1]
 [2425    0    1]
 [2495    0    2]
 [2535    0    2]
 [2555    0    1]
 [2580    0    2]
 [2585    0    1]
 [2635    0    1]
 [2655    0    1]
 [2670    0    2]
 [2670    0    2]
 [2695    0    2]
 [2720    0    1]
 [2740    0    2]
 [2745    0    1]
 [2745    0    2]
 [2750    0    2]
 [2755    0    1]
 [2800    0    2]
 [2805    0    1]
 [2810    0    1]
 [2810    0    2]
 [2824    0    1]
 [2824    0    2]
 [2834    0    2]
 [2839    0    1]
 [2839    0    1]
 [2839    0    1]
 [2849    0    1]
 [2875    0    1]
 [2885    0    1]
 [2900    0    2]
 [2915    0    1]
 [2990    0    2]
 [3160    0    2]
 [3160    0    2]
 [3180    0    1]
 [3210    0    1]
 [3230    0    1]
 [3250    0    1]
 [3260    0    1]
 [3260    0    2]
 [3270    0    2]
 [3285    0    1]
 [3300    0    1]
 [3305    0    1]
 [3310    0    1]
 [3310    0    2]
 [3310    0    2]
 [3360    0    1]
 [3385    0    2]
 [3390    0    2]
 [3390    0    2]
 [3400    0    2]
 [3435    0    2]
 [3439    0    1]
 [3439    0    1]
 [3439    0    1]
 [3459    0    2]
 [3469    0    2]
 [3484    0    2]
 [3489    0    1]
 [3500    0    2]
 [3500    0    2]
 [3525    0    2]
 [3545    0    1]
 [3545    0    2]
 [3555    0    1]
 [3555    0    2]
 [3565    0    2]
 [3570    0    1]
 [3580    0    2]
 [3600    0    2]
 [3605    0    1]
 [3615    0    1]
 [3665    0    2]
 [3705    0    2]
 [3780    0    1]
 [3785    0    2]
 [3825    0    1]
 [3870    0    2]
 [3895    0    1]
 [3920    0    1]
 [3950    0    1]
 [3950    0    2]
 [3955    0    1]
 [3955    0    2]
 [3955    0    2]
 [3960    0    1]
 [3965    0    1]
 [3985    0    2]
 [3995    0    2]
 [4015    0    2]
 [4040    0    1]
 [4045    0    2]
 [4060    0    1]
 [4094    0    1]
 [4094    0    2]
 [4094    0    2]
 [4100    0    1]
 [4105    0    2]
 [4110    0    1]
 [4110    0    2]
 [4110    0    2]
 [4110    0    2]
 [4120    0    1]
 [4160    0    1]
 [4170    0    2]
 [4185    0    1]
 [4190    0    1]
 [4190    0    2]
 [4195    0    1]
 [4310    0    1]
 [4335    0    1]
 [4335    0    1]
 [4360    0    2]
 [4370    0    2]
 [4520    0    1]]
Backend qtagg is interactive backend. Turning interactive mode on.

