"""Drift monitoring: PSI on score distributions (+ optional Evidently HTML report)."""
import numpy as np


def psi(ref, cur, bins: int = 10) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    if len(ref) < 5 or len(cur) < 5:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def drift_status(value: float) -> str:
    return "retrain recommended" if value > 0.25 else "watch" if value > 0.1 else "stable"


def evidently_report(ref, cur, out="reports/drift.html"):
    """Writes an Evidently report if the package is installed."""
    try:
        import os
        import pandas as pd
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
        os.makedirs(os.path.dirname(out), exist_ok=True)
        rep = Report(metrics=[DataDriftPreset()])
        rep.run(reference_data=pd.DataFrame({"p_fake": ref}), current_data=pd.DataFrame({"p_fake": cur}))
        rep.save_html(out)
        return out
    except Exception:
        return None
