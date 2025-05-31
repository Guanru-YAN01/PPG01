from pathlib import Path
import pandas as pd, numpy as np, pickle
from scipy.signal import butter, filtfilt, detrend

# Session category map
SESSION_CATEGORY = {
    'rest-1':'Still','rest-2':'Still','rest-3':'Still','rest-4':'Still','rest-5':'Still',
    'meditation-1':'Still','meditation-2':'Still',
    'screen-reading':'Still',
    'standing':'Still',
    'walking':'Rhythmic movement',
    'jogging':'Rhythmic movement',
    'running':'Rhythmic movement',
    'keyboard-typing':'Non-rhythmic movement',
    'mobile-typing':'Non-rhythmic movement',
    'ssst-sing':'Non-rhythmic movement'
}

# Parameters
FS = 64                   # Sampling rate (Hz)
WIN_SIZE = 8 * FS         # 8-second window = 512 samples
STEP_SIZE = 2 * FS        # 2-second step = 128 samples
INPUT = Path('original_galaxy')
OUTPUT = Path('clean_galaxy')
OUTPUT.mkdir(parents=True, exist_ok=True)

# Basic signal processing
def butter_lowpass_filter(data, cutoff=6.0, fs=64.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def preprocess_bvp(x):
    x = detrend(x)  # Linear detrending
    baseline = np.convolve(x, np.ones(FS)/FS, mode='same')  # baseline drift (1s mean)
    x = x - baseline
    x = butter_lowpass_filter(x, cutoff=6.0, fs=FS)
    return x

# Loaders
def load_csv(p):
    df = pd.read_csv(p)
    df.timestamp /= 1e6  # µs → s
    return df

def extract_intervals(ev):
    ev.columns = ['timestamp','session','status']
    enter, ivs = {}, []
    for ts, sid, st in ev.itertuples(False):
        if st == 'ENTER':
            enter[sid] = ts
        elif st == 'EXIT' and sid in enter:
            if sid in SESSION_CATEGORY:
                ivs.append((sid, enter.pop(sid), ts))
    return ivs

# Main processing function
def process(pid):
    print(f"\n=== Processing {pid} ===")
    base = INPUT / pid / 'E4'
    if not base.exists():
        print(f"  ! E4 folder not found at {base}")
        return

    bvp = load_csv(base / 'BVP.csv')
    acc = load_csv(base / 'ACC.csv')
    hr  = load_csv(base / 'HR.csv')
    ev  = pd.read_csv(INPUT / pid / 'Event.csv')
    ev['timestamp'] = ev['timestamp'] / 1e3  # µs → s
    ev = ev[ev['session'].isin(SESSION_CATEGORY.keys())]
    ivs = extract_intervals(ev)

    data = {}
    for sid, t0, t1 in ivs:
        mask = lambda df: df.timestamp.between(t0, t1)
        b = bvp[mask(bvp)].value.to_numpy()
        a = acc[mask(acc)][['x','y','z']].to_numpy()
        t = bvp[mask(bvp)].timestamp.to_numpy()
        if b.size < WIN_SIZE or a.shape[0] < WIN_SIZE:
            continue

        b = preprocess_bvp(b)
        hr_win = hr[hr.timestamp.between(t0-4, t1+4)]

        for i in range(0, min(len(b), len(a)) - WIN_SIZE + 1, STEP_SIZE):
            h = hr_win[hr_win.timestamp.between(t[i], t[i+WIN_SIZE-1])].value.to_numpy()
            if not h.size:
                continue

            data.setdefault(sid, []).append({
                'ppg': b[i:i+WIN_SIZE],
                'acc': a[i:i+WIN_SIZE],
                'hr' : float(h.mean())
            })

    for sid, samples in data.items():
        print(f"  → Session {sid}: {len(samples)} windows")
        cat = SESSION_CATEGORY[sid]
        out = OUTPUT / pid / cat
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f'{sid}.pkl', 'wb') as f:
            pickle.dump(samples, f)

if __name__ == '__main__':
    parts = [d.name for d in INPUT.iterdir() if d.name.startswith('P')]
    for pid in parts:
        process(pid)
    print("\nAll done.")

'''
.pkl is list of dict, for every dict:
{
    'bvp': np.ndarray,           # shape (512,), BVP after basic preprocessing
    'acc': np.ndarray,           # shape (256, 3), raw ACC (I don't it now, so I leave it raw)
    'hr': float,                 # HR value for every windowe (bpm)       
}

'''