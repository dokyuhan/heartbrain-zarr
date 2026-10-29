import numpy as np
from scipy import signal
import vollmer_hrv as vh


def compute_window_starts(n_samples, fs, window_sec, overlap):
    """Window boundaries used for EEG and the cardiac source(s), so index i refers to the same samples in every array."""
    win = int(round(window_sec * fs))
    step = int(round(win * (1 - overlap)))
    if n_samples < win:
        return win, []
    return win, list(range(0, n_samples - win + 1, step))


def welch_psd(segment, welch):
    """Welch PSD of one window for all channels: (n_channels, n_samples) -> (n_channels, n_freq), sliced to 0..fmax.
    Units: (input signal units)^2 / Hz. A channel that is non-finite or flat in this window gets an all-NaN row, the others are unaffected."""
    out = np.full((segment.shape[0], welch.n_freq), np.nan)
    ok = np.all(np.isfinite(segment), axis=1) & (np.ptp(segment, axis=1) > 0)
    if ok.any():
        f, psd = signal.welch(segment[ok], fs=welch.fs, window=welch.window, nperseg=welch.nperseg, noverlap=welch.noverlap,
                              detrend=welch.detrend, scaling="density", average="mean", axis=-1)
        assert len(f) == len(welch.freq_full_hz) and np.allclose(f, welch.freq_full_hz)
        out[ok] = psd[:, :welch.n_freq]
    return out


def add_gate_flags(quality, gate_thresholds, min_pairs):
    """passes_gate_<name> = valid_share >= threshold and n_pairs >= min_pairs (int8). Study choice: the caller supplies the thresholds."""
    for name, th in gate_thresholds.items():
        quality[f"passes_gate_{name}"] = ((quality["valid_share"] >= th) & (quality["n_pairs"] >= min_pairs)).astype(np.int8)
    return quality


def detect_ecg_sources(ecg_signals, cfg):
    """{source: 1-D signal} -> {source: (beats, rr)} by whole-segment detection (vollmer_hrv). Parameters come from cfg.ecg."""
    e = cfg.ecg
    return {src: vh.detect_beats_and_rr(sig, cfg.fs, e.beat_min_bpm, e.beat_max_bpm, e.rr_limit_pct) for src, sig in ecg_signals.items()}


def extract_window_set(eeg, ecg_beats, window_sec, cfg):
    """All windows of one segment at one window length: EEG PSD + each cardiac source over identical boundaries.
    eeg: (n_channels, n_samples); ecg_beats: {source: (beats, rr)} from detect_ecg_sources on a signal of the same length."""
    n_samples = eeg.shape[1]
    win, starts = compute_window_starts(n_samples, cfg.fs, window_sec, cfg.windows.overlap)
    n = len(starts)
    start_sample = np.array(starts, dtype=np.int64)
    end_sample = start_sample + win
    psd = np.full((n, eeg.shape[0], cfg.welch.n_freq), np.nan)
    for i, s in enumerate(starts):
        psd[i] = welch_psd(eeg[:, s:s + win], cfg.welch)
    ecg_out = {}
    for src, (beats, rr) in ecg_beats.items():
        r = vh.hrv_windows_from_beats(beats, rr, starts, win, cfg.ecg.min_pairs)
        add_gate_flags(r["quality"], cfg.ecg.gate_thresholds, cfg.ecg.min_pairs)
        ecg_out[src] = r
    return {"window_index": np.arange(n, dtype=np.int64), "start_sample": start_sample, "end_sample": end_sample,
            "start_time": start_sample / cfg.fs, "end_time": end_sample / cfg.fs, "psd": psd, "ecg": ecg_out,
            "window_samples": win, "step_samples": int(round(win * (1 - cfg.windows.overlap)))}
