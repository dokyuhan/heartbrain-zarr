import vollmer_hrv as vh

def write_window_group(parent_grp, window_sec, result, cfg):
    w = parent_grp.create_group(f"windows_{window_sec}s")
    w.attrs.update({"window_seconds": window_sec, "window_samples": int(result["window_samples"]),
                    "step_samples": int(result["step_samples"]), "overlap": cfg.windows.overlap,
                    "n_windows": int(len(result["window_index"]))})

    timing = w.create_group("timing")
    for key in ["window_index", "start_sample", "end_sample", "start_time", "end_time"]:
        timing.create_array(key, data=result[key])
    timing["start_time"].attrs["units"] = "s"
    timing["end_time"].attrs["units"] = "s"
    timing["end_sample"].attrs["description"] = "exclusive end: window covers samples [start_sample, end_sample)"

    wc = cfg.welch
    eeg = w.create_group("eeg")
    eeg.attrs.update({
        "method": "Welch", "nperseg": wc.nperseg, "noverlap": wc.noverlap, "window": wc.window,
        "detrend": wc.detrend, "scaling": "density", "average": "mean",
        "frequency_resolution_hz": wc.resolution_hz, "fmin_hz": float(wc.frequency_hz[0]),
        "fmax_hz": float(wc.frequency_hz[-1]), "psd_dims": ["window", "channel", "frequency"],
        "channels": list(cfg.eeg_channels), "psd_units": cfg.psd_units,
    })
    eeg.create_array("frequency_hz", data=wc.frequency_hz)
    eeg["frequency_hz"].attrs["units"] = "Hz"
    eeg.create_array("psd", data=result["psd"])

    e = cfg.ecg
    for src, ecg_res in result["ecg"].items():
        ecg = w.create_group(src)
        attrs = {"processor": e.processor, "sample_rate": cfg.fs, "method": e.method,
                 "beat_range_bpm": [e.beat_min_bpm, e.beat_max_bpm], "rr_filter_limit_pct": e.rr_limit_pct, "min_pairs": e.min_pairs}
        if e.sources.get(src) is not None:
            attrs["source"] = e.sources[src]
        ecg.attrs.update(attrs)
        metrics = ecg.create_group("metrics")
        for key, arr in ecg_res["metrics"].items():
            metrics.create_array(key, data=arr)
            metrics[key].attrs.update(vh.ECG_METRIC_ATTRS[key])
        derived = ecg.create_group("derived")
        derived.create_array("ln_rmssd", data=ecg_res["derived"]["ln_rmssd"])
        derived["ln_rmssd"].attrs.update({"transformation": "natural logarithm of rmssd", "note": e.ln_rmssd_note})
        quality = ecg.create_group("quality")
        quality.attrs.update(e.quality_attrs)
        for key, arr in ecg_res["quality"].items():
            quality.create_array(key, data=arr)
            quality[key].attrs.update(vh.ECG_QUALITY_ATTRS.get(key) or e.extra_quality_attrs.get(key, {}))
