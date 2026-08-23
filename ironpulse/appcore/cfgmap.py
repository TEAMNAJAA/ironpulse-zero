BAND_NAMES = [("sub", "E_sub"), ("one_x", "E_1x"), ("mid", "E_mid"), ("high", "E_hi")]


def rbp_blade_count(app_cfg, blade_count):
    if not blade_count:
        return None
    need = int(app_cfg["machine"]["rbp_min_blade_count"])
    return blade_count if int(blade_count) >= need else None


def core_cfg(app_cfg, fs, f0_hz, blade_count=None):
    a = app_cfg["analysis"]
    nyq_order = (float(fs) / 2.0) / float(f0_hz)
    bands = {}
    for app_key, core_key in BAND_NAMES:
        lo, hi = a["bands"][app_key]
        hi = nyq_order if hi is None else min(float(hi), nyq_order)
        bands[core_key] = [float(lo), float(hi)]
    return {
        "seed": int(app_cfg["model"]["random_state"]),
        "signal": {
            "detrend": int(a["detrend_order"]),
            "window_fn": a["window_fn"],
            "window_sec": float(a["window_sec"]),
            "hop_sec": float(a["hop_sec"]),
        },
        "features": {
            "peak_search_halfwidth_order": float(a["order_tolerance"]),
            "bands": bands,
            "max_order": nyq_order,
            "blade_count": blade_count,
        },
        "integration": {
            "highpass_order_floor": float(a["rms_order_min"]),
        },
        "detectors": {
            "pca_variance": float(app_cfg["model"]["mahalanobis"]["pca_variance"]),
            "threshold_percentile": float(app_cfg["model"]["threshold_percentile"]),
            "one_class_svm": {
                "nu": float(app_cfg["model"]["ocsvm"]["nu"]),
                "gamma": app_cfg["model"]["ocsvm"]["gamma"],
            },
            "isolation_forest": {
                "n_estimators": int(app_cfg["model"]["isolation_forest"]["n_estimators"]),
                "max_samples": app_cfg["model"]["isolation_forest"]["max_samples"],
            },
            "pca_reconstruction": {
                "variance": float(app_cfg["model"]["mahalanobis"]["pca_variance"]),
            },
        },
        "evaluate": {"n_folds": 5, "group_by": "file_id"},
    }
