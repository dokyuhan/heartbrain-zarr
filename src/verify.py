import numpy as np
import zarr
import vollmer_hrv as vh
from .features import compute_window_starts

def walk(group, prefix=""):
    """Yield (path, node) for every group/array below `group`."""
    for name, node in group.members():
        path = f"{prefix}/{name}" if prefix else name
        yield path, node
        if isinstance(node, zarr.Group):
            yield from walk(node, path)

def check_forbidden_names(root, forbidden_parts, forbidden_names=(), forbidden_prefixes=(), meta_group="metadata"):
    """No baked-in band definitions / gate-style arrays: checks node names and attribute keys (and the metadata keys for `forbidden_parts`)."""
    problems = []
    for path, node in walk(root):
        for nm in [path.split("/")[-1].lower()] + [str(k).lower() for k in node.attrs]:
            if any(part in nm for part in forbidden_parts) or nm in forbidden_names or (forbidden_prefixes and nm.startswith(tuple(forbidden_prefixes))):
                problems.append(f"F/G: forbidden name '{nm}' at {path}")
    for k in dict(root[meta_group].attrs):
        if any(part in str(k).lower() for part in forbidden_parts):
            problems.append(f"F/G: forbidden metadata key {k}")
    return problems

def verify_window_group(w, window_sec, tag, cfg, n_samples=None):
    """Checks one windows_<N>s group: alignment of all arrays (C), window length/count/index, EEG dimensions (D), frequency vector (E),
    quality-field consistency incl. the gate flags (I), even-index windows == 0 % overlap windows (H, only for 50 % overlap)."""
    problems = []
    wi, ss, es = w["timing/window_index"][:], w["timing/start_sample"][:], w["timing/end_sample"][:]
    psd, fhz = w["eeg/psd"], w["eeg/frequency_hz"][:]
    n = psd.shape[0]
    lens = {"window_index": len(wi), "start_sample": len(ss), "end_sample": len(es), "psd": n}
    srcs = list(cfg.ecg.sources)
    for src in srcs:
        if src not in w:
            problems.append(f"A: missing {tag}/{src}"); continue
        lens.update({f"{src}/metrics/{k}": w[f"{src}/metrics/{k}"].shape[0] for k in vh.ECG_METRIC_KEYS})
        lens[f"{src}/ln_rmssd"] = w[f"{src}/derived/ln_rmssd"].shape[0]
        lens.update({f"{src}/quality/{k}": w[f"{src}/quality/{k}"].shape[0] for k in vh.ECG_QUALITY_KEYS})
        lens.update({f"{src}/quality/passes_gate_{g}": w[f"{src}/quality/passes_gate_{g}"].shape[0] for g in cfg.ecg.gate_thresholds})
    if len(set(lens.values())) != 1:
        problems.append(f"C: length mismatch in {tag}: {lens}")
    mp = cfg.ecg.min_pairs
    for src in srcs:                                              # I. quality fields are consistent with each other and with the measures
        if src not in w:
            continue
        qd = {k: w[f"{src}/quality/{k}"][:] for k in vh.ECG_QUALITY_KEYS}
        rm = w[f"{src}/metrics/rmssd"][:]
        if np.any(qd["n_valid_rr"] > qd["n_rr"]) or np.any(qd["n_pairs"] > np.maximum(qd["n_valid_rr"] - 1, 0)):
            problems.append(f"I: n_valid_rr / n_pairs inconsistent with n_rr in {tag}/{src}")
        if np.any((qd["valid_share"] < 0) | (qd["valid_share"] > 1)):
            problems.append(f"I: valid_share outside 0-1 in {tag}/{src}")
        for gname, th in cfg.ecg.gate_thresholds.items():
            expected = ((qd["valid_share"] >= th) & (qd["n_pairs"] >= mp)).astype(np.int8)
            if not np.array_equal(w[f"{src}/quality/passes_gate_{gname}"][:], expected):
                problems.append(f"I: passes_gate_{gname} does not match valid_share / n_pairs in {tag}/{src}")
        if np.any(np.isfinite(rm) & (qd["n_pairs"] < mp)) or np.any(~np.isfinite(rm) & (qd["n_pairs"] >= mp)):
            problems.append(f"I: rmssd is not finite exactly when n_pairs >= {mp} in {tag}/{src}")
    if not np.array_equal(wi, np.arange(n)):
        problems.append(f"C: window_index is not 0..N-1 in {tag}")
    if n and not np.all(es - ss == int(round(window_sec * cfg.fs))):
        problems.append(f"C: window length != {window_sec}s in {tag}")
    if n_samples is not None:
        if n and es.max() > n_samples:
            problems.append(f"C: a window of {tag} ends after the segment ({es.max()} > {n_samples})")
        _, expected_starts = compute_window_starts(n_samples, cfg.fs, window_sec, cfg.windows.overlap)
        if len(expected_starts) != n:
            problems.append(f"C: {tag} has {n} windows, recomputation from n_samples gives {len(expected_starts)}")
        if abs(cfg.windows.overlap - 0.5) < 1e-12:
            _, starts0 = compute_window_starts(n_samples, cfg.fs, window_sec, 0.0)
            if not np.array_equal(ss[::2], np.array(starts0, dtype=np.int64)):
                problems.append(f"H: even-index windows of {tag} are not the 0%-overlap windows")
    if psd.ndim != 3 or psd.shape[1] != len(cfg.eeg_channels) or psd.shape[2] != len(fhz):
        problems.append(f"D: bad psd shape {psd.shape} vs {len(fhz)} freqs in {tag}")
    if not np.all(np.diff(fhz) > 0) or not np.allclose(np.diff(fhz), cfg.welch.resolution_hz):
        problems.append(f"E: frequency_hz not increasing at {cfg.welch.resolution_hz} Hz in {tag}")
    return problems


def print_node_tree(group, title=None):
    """Full node listing (names, shapes, dtypes) of one group, e.g. one repetition or block."""
    if title:
        print(title)
    for path, node in walk(group):
        indent = "  " + "  " * path.count("/")
        if isinstance(node, zarr.Group):
            print(f"{indent}{path.split('/')[-1]}/")
        else:
            print(f"{indent}{path.split('/')[-1]}  {node.shape} {node.dtype}")
