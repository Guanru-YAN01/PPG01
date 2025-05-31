import os
import pickle
import numpy as np
from collections import defaultdict
from scipy.signal import detrend, butter, filtfilt

# Activity
activity_info = {
    1: ('Sitting', 'Still'),
    2: ('Stairs', 'Rhythmic movement'),
    3: ('Soccer', 'Non-rhythmic movement'),
    4: ('Cycling', 'Rhythmic movement'),
    5: ('Driving', 'Non-rhythmic movement'),
    6: ('Lunch', 'Non-rhythmic movement'),
    7: ('Walking', 'Rhythmic movement'),
    8: ('Working', 'Non-rhythmic movement')
}

# Sampling and window parameters
FS_BVP = 64          # BVP sample rate (Hz)
BVP_WIN = 512        # window size in samples (8s * 64Hz)
ACC_WIN = 256        # ACC window size (8s * 32Hz)
STEP_BVP = 128       # step size (2s * 64Hz)
STEP_ACC = 64        # ACC step (2s * 32Hz)

# Remove linear trend using scipy
def detrend_signal(x: np.ndarray) -> np.ndarray:
    return detrend(x, type='linear')

# Subtract moving average baseline
def remove_baseline(x: np.ndarray, window_size: int) -> np.ndarray:
    x = np.asarray(x).flatten()
    kernel = np.ones(window_size) / window_size
    baseline = np.convolve(x, kernel, mode='same')
    return x - baseline

# Apply Butterworth bandpass filter between cutoff_low and cutoff_high
def butterworth_bandpass(x: np.ndarray,
                         cutoff_low: float,
                         cutoff_high: float,
                         fs: float,
                         order: int = 3) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(order, [cutoff_low / nyq, cutoff_high / nyq], btype='band')
    return filtfilt(b, a, x)

# Window extraction
def extract_windows(signal: np.ndarray, window_size: int, step: int) -> np.ndarray:
    count = (len(signal) - window_size) // step + 1
    return np.stack([signal[i*step : i*step + window_size]
                     for i in range(count)])

# Full-signal processing: detrend, remove baseline, bandpass filter
def preprocess_bvp(bvp_raw: np.ndarray) -> np.ndarray:
    x = detrend_signal(bvp_raw)
    x = remove_baseline(x, window_size=FS_BVP)
    x = butterworth_bandpass(x, cutoff_low=0.5, cutoff_high=8.0, fs=FS_BVP)
    return x

# Processing per subject
def process_subject_file(file_path: str, output_base: str):
    subject_id = os.path.splitext(os.path.basename(file_path))[0]
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Extract raw signals and labels
    bvp_raw = np.asarray(data['signal']['wrist']['BVP'])
    acc_raw = np.asarray(data['signal']['wrist']['ACC'])
    hr_series = np.asarray(data['label'])
    activities = np.asarray(data['activity']).flatten()

    # Preprocess BVP globally
    bvp = preprocess_bvp(bvp_raw)

    # Segment into windows
    bvp_windows = extract_windows(bvp, BVP_WIN, STEP_BVP)
    acc_windows = extract_windows(acc_raw, ACC_WIN, STEP_ACC) if acc_raw.size else None

    # Prepare output directories
    for _, category in activity_info.values():
        os.makedirs(os.path.join(output_base, subject_id, category), exist_ok=True)

    # Collect samples per activity
    samples_by_activity = defaultdict(list)
    for i in range(len(bvp_windows)):
        # Determine activity for each window using ACC-based index
        start_idx = i * STEP_ACC
        end_idx = start_idx + ACC_WIN
        act_ids = activities[start_idx:end_idx]
        if act_ids.size == 0:
            continue
        activity_id = int(np.round(np.mean(act_ids)))
        if activity_id not in activity_info:
            continue
        name, category = activity_info[activity_id]

        sample = {
            'bvp': bvp_windows[i],
            'acc': acc_windows[i] if acc_windows is not None else None,
            'hr': float(hr_series[i]),
            'activity_id': activity_id,
            'activity_name': name
        }
        samples_by_activity[(name, category)].append(sample)

    # Save samples grouped by activity
    for (name, category), samples in samples_by_activity.items():
        out_dir = os.path.join(output_base, subject_id, category)
        out_path = os.path.join(out_dir, f"{name}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(samples, f)

# Main function
def main(input_root: str = 'original_dalia', output_root: str = 'clean_dalia'):
    for fname in sorted(os.listdir(input_root)):
        if not fname.endswith('.pkl'):
            continue
        process_subject_file(os.path.join(input_root, fname), output_root)

if __name__ == '__main__':
    main()


'''
.pkl is list of dict, for every dict:
{
    'bvp': np.ndarray,           # shape (512,), BVP after basic preprocessing
    'acc': np.ndarray,           # shape (256, 3), raw ACC (I don't it now, so I leave it raw)
    'hr': float,                 # HR value for every windowe (bpm)
    'activity_id': int,          
    'activity_name': str         
}

'''