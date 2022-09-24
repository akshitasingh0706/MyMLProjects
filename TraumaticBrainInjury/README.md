# Electroencephalograph (EEG)

## Getting Data
library: MNE

## Preprocessing
### Independent component Analysis (ICA)
EEG data can easily be contaminated by human activity such eye movements, drowsiness etc. ICA can effectively detect and separate activity from various artifacts. 
Approach:
ICA is a signal processing method that can separate independent sources linearly mixed in several sensors. For EEGs specifically,  ICA can separate out artifacts embedded in the data (since they are usually independent of each other).
An important step in ICA is known as Whitening. This just means that we transform it in such a way that potential correlations between its components are removed (covariance equal to 0) and the variance of each component is equal to 1. Covariance matrix of whitened signal = Idenity matrix. 

### Parameters
Frequency Bands: EEG waveforms can be classified into 5 frequency bands - 1) delta (0.5-4 Hz), 2) theta (4-8 Hz) , 3) alpha (8-14), 4) beta (14-30), and 5) gamma (over 30 Hz)

Sleep Stages: Sleep occurs in five stages: wake, N1, N2, N3, and REM. Stages N1 to N3 are considered non-rapid eye movement (NREM) sleep, with each stage a progressively deeper sleep. 

Channels: An electrode capturing brainwave activity is called an EEG channel. Typical EEG systems can have as few as a single channel to as many as 256 channels

Epoch Length: EEG epoching is a procedure in which specific time-windows are extracted from the continuous EEG signal. These time windows are called “epochs”, and usually are time-locked with respect an event e.g. a visual stimulus.

Amplitude: the strength of the pattern in terms of microvolts of electrical energy

Absolute vs Relative power: EEG spectral power can be expressed in absolute form (energy in a chosen frequency band) or in relative form (energy in a chosen frequency band divided by the total energy from all of the frequency bands). Here power refers to Power Spectral Density (PSD).

Spectral entropy: a signal is a measure of its spectral power distribution. (related to Shannon Entropy)

Sleep spindles: pattern of brain waves people experience during certain stages of sleep. Sleep spindles occur in the midst of slow-wave sleep, hence they are easy to identify due to their increased frequency. 

Cross-Frequency Coupling: Interaction between oscillations at different frequency bands. Types: 1) Phase-phase coupling, 2) Amplitude-amplitude coupling, and 3) Phase-amplitude coupling

Coherence: a measure of the variability of time differences between two time series in a specific frequency band. The Fourier transform provides a direct relationship between the time and frequency domains and represents time difference as a phase difference or phase angle.

Hjorth parameters: indicators of statistical properties used in signal processing in the time domain. The parameters are 1) Activity (represents the signal power, the variance of a time function), 2) Mobility (represents the mean frequency or the proportion of standard deviation of the power spectrum), and 3) Complexity ( represents the change in frequency).The parameters are normalised slope descriptors (NSDs) used in EEG

### TBI Questionnaire Acronyms
PHQ-9: Patient Health Questionnaire
PCL-5: Posttraumatic stress disorder checklist
RPQ: Rivermead Post-Concussion Symptoms Questionnaire
ISI: Insomnia Severity Index


## Resources
1) ICA: https://arnauddelorme.com/ica_for_dummies/
2) ICA Algorithm: https://towardsdatascience.com/independent-component-analysis-ica-in-python-a0ef0db0955e
