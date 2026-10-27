
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

from scipy import signal
from scipy.signal import butter, sosfiltfilt

from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# 1. Project paths
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATASET_DIR = PROJECT_DIR / "data" / "ESC-50-master"
META_PATH = DATASET_DIR / "meta" / "esc50.csv"
AUDIO_DIR = DATASET_DIR / "audio"

DECISION_TREE_PATH = (
    PROJECT_DIR / "models" / "sound_guardian_model.joblib"
)

RANDOM_FOREST_PATH = (
    PROJECT_DIR / "models" / "best_sound_guardian_model.joblib"
)

OUTPUT_PATH = (
    PROJECT_DIR / "results" / "dsp_robustness_comparison.csv"
)

SUMMARY_PATH = (
    PROJECT_DIR / "results" / "dsp_robustness_summary.csv"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Load both trained models
# ==========================================

for path in [
    META_PATH,
    DECISION_TREE_PATH,
    RANDOM_FOREST_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

decision_tree_saved = joblib.load(DECISION_TREE_PATH)
random_forest_saved = joblib.load(RANDOM_FOREST_PATH)

models = {
    "Decision Tree": decision_tree_saved["model"],
    "Random Forest": random_forest_saved["model"],
}

feature_columns = decision_tree_saved["feature_columns"]

if feature_columns != random_forest_saved["feature_columns"]:
    raise ValueError(
        "The two models use different feature columns."
    )

print("=" * 70)
print("HOME SOUND GUARDIAN - DSP ROBUSTNESS COMPARISON")
print("=" * 70)

print("\nLoaded models:")
for model_name in models:
    print("-", model_name)


# ==========================================
# 3. Load the official test fold
# ==========================================

classes = [
    "dog",
    "crying_baby",
    "door_wood_knock",
    "clock_alarm",
    "glass_breaking",
    "vacuum_cleaner",
    "rain",
    "thunderstorm",
]

meta = pd.read_csv(META_PATH)

test_meta = meta[
    (meta["fold"] == 5)
    & (meta["category"].isin(classes))
].copy()

test_meta = test_meta.reset_index(drop=True)

if len(test_meta) == 0:
    raise ValueError("No test recordings found in fold 5.")

if set(test_meta["category"].unique()) != set(classes):
    raise ValueError("Some expected classes are missing from test fold.")

print(f"\nTest recordings: {len(test_meta)}")
print("Expected test recordings: 64")

print("\nTest recordings per class:")
print(test_meta["category"].value_counts().sort_index().to_string())


# ==========================================
# 4. Extract the same features as training
# ==========================================

def extract_features(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    rms = np.sqrt(np.mean(audio ** 2))

    zcr = np.mean(
        np.abs(np.diff(np.signbit(audio)))
    )

    frequencies, frame_times, power = signal.spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=2048,
        noverlap=1024,
        detrend=False,
        scaling="spectrum",
        mode="psd",
    )

    power_sum = np.sum(power, axis=0) + 1e-12

    centroid = np.sum(
        frequencies[:, None] * power,
        axis=0,
    ) / power_sum

    bandwidth = np.sqrt(
        np.sum(
            ((frequencies[:, None] - centroid) ** 2) * power,
            axis=0,
        ) / power_sum
    )

    cumulative_power = np.cumsum(power, axis=0)
    threshold = 0.85 * cumulative_power[-1, :]

    rolloff_indices = np.argmax(
        cumulative_power >= threshold[None, :],
        axis=0,
    )

    rolloff = frequencies[rolloff_indices]

    return {
        "rms": float(rms),
        "zcr": float(zcr),
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "bandwidth_std": float(np.std(bandwidth)),
        "rolloff_mean": float(np.mean(rolloff)),
        "rolloff_std": float(np.std(rolloff)),
    }


# ==========================================
# 5. Add white noise at a specified SNR
# ==========================================

def add_noise(audio, snr_db, rng):
    signal_power = np.mean(audio ** 2)

    if signal_power < 1e-12:
        return audio.copy()

    noise = rng.normal(size=audio.shape)
    noise_power = np.mean(noise ** 2)

    target_noise_power = (
        signal_power / (10 ** (snr_db / 10))
    )

    noise *= np.sqrt(target_noise_power / noise_power)

    return audio + noise


# ==========================================
# 6. Apply Butterworth band-pass filter
# ==========================================

def apply_bandpass(audio, sample_rate):
    low_cutoff = 100.0
    high_cutoff = min(8000.0, sample_rate * 0.45)

    if low_cutoff >= high_cutoff:
        raise ValueError("Invalid band-pass cutoff frequencies.")

    sos = butter(
        4,
        [low_cutoff, high_cutoff],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )

    return sosfiltfilt(sos, audio)


# ==========================================
# 7. Generate each audio condition once
# ==========================================

conditions = [
    "Clean",
    "Noise_20dB",
    "Noise_10dB",
    "Noise_0dB",
    "Bandpass_Filter",
]

# Fixed seed makes the experiment reproducible.
rng = np.random.default_rng(42)

true_labels = []
features_by_condition = {
    condition: [] for condition in conditions
}

print("\nExtracting features for all test conditions...")

for i, row in enumerate(test_meta.itertuples(index=False), start=1):
    audio_path = AUDIO_DIR / row.filename

    audio, sample_rate = sf.read(audio_path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float64)

    if sample_rate != 44100:
        raise ValueError(
            f"Unexpected sample rate for {row.filename}: {sample_rate}"
        )

    true_labels.append(row.category)

    # Create each variant once so both models
    # receive exactly the same processed recordings.
    variants = {
        "Clean": audio,
        "Noise_20dB": add_noise(audio, 20, rng),
        "Noise_10dB": add_noise(audio, 10, rng),
        "Noise_0dB": add_noise(audio, 0, rng),
        "Bandpass_Filter": apply_bandpass(audio, sample_rate),
    }

    for condition, variant in variants.items():
        feature_values = extract_features(
            variant,
            sample_rate,
        )

        features_by_condition[condition].append(
            feature_values
        )

    if i % 10 == 0 or i == len(test_meta):
        print(f"Processed {i}/{len(test_meta)} recordings")


# ==========================================
# 8. Evaluate both models under every condition
# ==========================================

results = []

for model_name, model in models.items():
    print("\n" + "=" * 70)
    print(f"Evaluating: {model_name}")
    print("=" * 70)

    for condition in conditions:
        X_condition = pd.DataFrame(
            features_by_condition[condition],
            columns=feature_columns,
        )

        y_pred = model.predict(X_condition)

        accuracy = accuracy_score(true_labels, y_pred)

        macro_f1 = f1_score(
            true_labels,
            y_pred,
            labels=classes,
            average="macro",
            zero_division=0,
        )

        results.append({
            "model": model_name,
            "condition": condition,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })

        print(
            f"{condition:18s} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro F1: {macro_f1:.4f}"
        )


# ==========================================
# 9. Save detailed results and summary
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_PATH, index=False)

summary_df = results_df.pivot(
    index="condition",
    columns="model",
    values=["accuracy", "macro_f1"],
)

summary_df.columns = [
    f"{metric}_{model_name.lower().replace(' ', '_')}"
    for metric, model_name in summary_df.columns
]

summary_df = summary_df.reset_index()
summary_df.to_csv(SUMMARY_PATH, index=False)

print("\n" + "=" * 70)
print("DETAILED DSP ROBUSTNESS RESULTS")
print("=" * 70)
print(results_df.to_string(index=False))

print("\n" + "=" * 70)
print("SIDE-BY-SIDE SUMMARY")
print("=" * 70)
print(summary_df.to_string(index=False))

print("\nDetailed results saved to:")
print(OUTPUT_PATH)

print("\nSide-by-side summary saved to:")
print(SUMMARY_PATH)

print("\nDSP robustness comparison completed successfully!")
