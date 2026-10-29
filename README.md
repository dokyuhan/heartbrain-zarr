# eegecg_zarr  (placeholder name)

Dataset-agnostic building blocks for turning one EEG + ECG recording into windowed features stored in a Zarr store:
Welch PSD per EEG channel and window, HRV measures per window (via `vollmer_hrv`), the writer for `windows_<N>s/{timing,eeg,<ecg>/{metrics,derived,quality}}`, and a verifier.

**No study parameters live in this package.** Sampling rate, channels, Welch settings, window lengths, overlap, RR-filter settings, quality-gate thresholds and
all descriptive text are passed in through the config objects in `config.py` (no defaults). Dataset-specific parts (reading the raw files, the
hierarchy of groups, segmentation, behaviour, metadata) stay in the dataset's own extraction notebook.

Status: under construction.
