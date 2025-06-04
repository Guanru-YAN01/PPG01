import logging
from typing import Sequence, Union
import numpy as np
import heartpy as hp
from scipy.fft import rfft, rfftfreq
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)

def estimate_hr_time_domain(ppg_signal: Union[np.ndarray, Sequence[float]], sample_rate: int = 64) -> float:
    try:
        wd, metrics = hp.process(ppg_signal, sample_rate, clean_rr=True)
        bpm = metrics.get('bpm', np.nan)
        if np.isnan(bpm):
            _logger.warning("HeartPy failed to detect valid peaks; returning NaN")
        return float(bpm)
    except Exception as e:
        _logger.warning(f"HeartPy processing error ({e}); returning NaN")
        return np.nan


def estimate_hr_frequency_domain(ppg_signal: Union[np.ndarray, Sequence[float]], sample_rate: int = 64) -> float:
    try:
        arr = np.asarray(ppg_signal, dtype=float)
        n = arr.size
        if n < 2:
            _logger.warning("Signal length < 2; FFT not possible; returning NaN")
            return np.nan

        centered = arr - arr.mean()
        freqs = rfftfreq(n, d=1.0 / sample_rate)
        amps = np.abs(rfft(centered))

        mask = (freqs >= 0.8) & (freqs <= 3.0) #search peak in 0.8–3.0 Hz (48–180 bpm)
        if not mask.any():
            _logger.warning("FFT frequency mask empty; returning NaN")
            return np.nan

        peak_idx = np.argmax(amps[mask])
        peak_freq = freqs[mask][peak_idx]
        return float(peak_freq * 60.0)
    except Exception as e:
        _logger.warning(f"FFT estimation error ({e}); returning NaN")
        return np.nan


def compute_metrics(y_true: Union[np.ndarray, Sequence[float]], y_pred: Union[np.ndarray, Sequence[float]]) -> dict:
    # convert to arrays
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))

    # avoid division by zero in MAPE
    nz = y_true_arr != 0
    mape = np.mean(np.abs((y_true_arr[nz] - y_pred_arr[nz]) / y_true_arr[nz])) * 100.0 if nz.any() else np.nan

    r2 = r2_score(y_true_arr, y_pred_arr)

    # Pearson r if variance exists
    if y_true_arr.size >= 2 and y_true_arr.std() > 0 and y_pred_arr.std() > 0:
        corr, _ = pearsonr(y_true_arr, y_pred_arr)
    else:
        corr = np.nan

    within_5 = np.abs(y_true_arr - y_pred_arr) <= 5.0
    pct_within_5 = within_5.mean() * 100.0

    return {
        'MAE': round(float(mae), 2),
        'RMSE': round(float(rmse), 2),
        'MAPE': round(float(mape), 2) if not np.isnan(mape) else np.nan,
        'R2': round(float(r2), 2),
        'Pearson r': round(float(corr), 2) if not np.isnan(corr) else np.nan,
        '±5bpm (%)': round(float(pct_within_5), 1),
    }


def check_nan_ratio(values: Union[np.ndarray, Sequence[float]]) -> float:
    # return percentage of NaNs in the sequence
    arr = np.asarray(values, dtype=float)
    total = arr.size
    if total == 0:
        return 100.00
    nan_count = int(np.isnan(arr).sum())
    return round((nan_count / total) * 100.0, 2)
