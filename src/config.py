"""Configuration objects. Every field is REQUIRED (no defaults): the study parameters belong to the calling notebook, not to the package."""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class WelchConfig:
    fs: int
    nperseg: int
    window: str
    detrend: str
    noverlap: int
    fmax_hz: float

    @property
    def freq_full_hz(self):
        return np.fft.rfftfreq(self.nperseg, d=1 / self.fs)

    @property
    def frequency_hz(self):
        f = self.freq_full_hz
        return f[f <= self.fmax_hz + 1e-9]

    @property
    def n_freq(self):
        return len(self.frequency_hz)

    @property
    def resolution_hz(self):
        return self.fs / self.nperseg


@dataclass(frozen=True)
class WindowConfig:
    lengths_sec: tuple
    overlap: float


@dataclass(frozen=True)
class EcgConfig:
    """Parameters handed to ``vollmer_hrv`` plus the descriptive text written into the store."""
    beat_min_bpm: float
    beat_max_bpm: float
    rr_limit_pct: float
    min_pairs: int
    gate_thresholds: dict          
    sources: dict                 
    processor: str
    method: str
    quality_attrs: dict 
    ln_rmssd_note: str
    extra_quality_attrs: dict


@dataclass(frozen=True)
class StoreConfig:
    fs: int
    eeg_channels: tuple
    psd_units: str
    welch: WelchConfig
    windows: WindowConfig
    ecg: EcgConfig
