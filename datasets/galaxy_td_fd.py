import os
import pickle
import pandas as pd
from utils import (
    estimate_hr_time_domain,
    estimate_hr_frequency_domain,
    compute_metrics,
    check_nan_ratio
)

BASE_PATH = 'clean_galaxy'
SAMPLE_RATE = 64
VALID_METHODS = ['time', 'freq']


def evaluate_galaxy(method='time', save_csv=True, save_path=None):
    if method not in VALID_METHODS:
        raise ValueError("method must be 'time' or 'freq'")

    results = []

    for subject in sorted(os.listdir(BASE_PATH)):
        subj_path = os.path.join(BASE_PATH, subject)
        if not os.path.isdir(subj_path):
            continue

        for activity_category in os.listdir(subj_path):
            cat_path = os.path.join(subj_path, activity_category)
            if not os.path.isdir(cat_path):
                continue

            for file in os.listdir(cat_path):
                if not file.endswith('.pkl'):
                    continue

                file_path = os.path.join(cat_path, file)
                activity_name = os.path.splitext(file)[0]

                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                gt_hr = []
                pred_hr = []

                for item in data:
                    ppg = item['ppg']
                    gt = item['hr']

                    if method == 'time':
                        pred = estimate_hr_time_domain(ppg, SAMPLE_RATE)
                    else:
                        pred = estimate_hr_frequency_domain(ppg, SAMPLE_RATE)

                    if not pd.isna(pred):
                        gt_hr.append(gt)
                        pred_hr.append(pred)

                if len(gt_hr) >= 3:
                    metrics = compute_metrics(gt_hr, pred_hr)
                    nan_pct = check_nan_ratio([
                        estimate_hr_time_domain(d['ppg'], SAMPLE_RATE) if method == 'time'
                        else estimate_hr_frequency_domain(d['ppg'], SAMPLE_RATE)
                        for d in data
                    ])
                    results.append({
                        'Subject': subject,
                        'Activity': activity_name,
                        **metrics,
                        'NaN (%)': nan_pct
                    })

    df = pd.DataFrame(results)
    if save_csv:
        if save_path is None:
            save_path = f'galaxy_{method}_domain_results.csv'
        df.to_csv(save_path, index=False)
        print(f"Saved results to {save_path}")

    return df


if __name__ == '__main__':
    evaluate_galaxy(method='time')
    evaluate_galaxy(method='freq')
