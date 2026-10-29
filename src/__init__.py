"""eegecg_zarr (placeholder name): windowed EEG PSD + ECG HRV written to Zarr; study parameters are passed in via config objects."""
from .config import WelchConfig, WindowConfig, EcgConfig, StoreConfig
from .features import compute_window_starts, welch_psd, add_gate_flags, detect_ecg_sources, extract_window_set
from .writer import write_window_group
from .verify import walk, check_forbidden_names, verify_window_group, print_node_tree

__version__ = "1.0.0"
