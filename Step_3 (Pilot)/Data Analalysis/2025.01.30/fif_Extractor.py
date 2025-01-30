import mne

file_path = r"H:\Post\6th_Phase (ECG-EEG Baced BCI)\2025.01.28\Pilot Data Analysis\Enhanced Epoching\Elmira\pe_sign_epochs\R1_Locked-prepro_epochs-epo.fif"
epochs = mne.read_epochs(file_path, preload=True)

print(epochs)
