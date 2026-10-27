
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from math import gcd


# ==========================================
# 1. Paths and configuration
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATASET_ROOT = PROJECT_DIR / "data" / "ESC-50-master"
META_PATH = DATASET_ROOT / "meta" / "esc50.csv"
AUDIO_DIR = DATASET_ROOT / "audio"
OUTPUT_PATH = PROJECT_DIR / "features.csv"

TARGET_SAMPLE_RATE = 44100

CLASSES = [
    "dog",
    "crying_baby",
    "door_wood_knock",
    "clock_alarm",
    "glass_breaking",
    "vacuum_cleaner",
    "rain",
    "thunderstorm",
]

NOISE_SNR_LEVELS = (20, 10)


# ==========================================
# 2. Load and preprocess audio
# ==========================================

def load_audio(audio_path, target_sr=TARGET_SAMPLE_RATE):
    audio, sr = sf.read(audio_path, dtype="float64")

    if audio.size == 0:
        raise ValueError(f"Empty audio: {audio_path}")

    if not np.isfinite(audio).all():
        raise ValueError(f"Invalid values in: {audio_path}")

    # Convert stereo/multichannel audio to mono.
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if audio.ndim != 1:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")

    # Resample consistently.
    if sr != target_sr:
        common = gcd(int(sr), int(target_sr))
        up = target_sr // common
        down = sr // common
        audio = signal.resample_poly(audio, up, down)
        sr = target_sr

    if len(audio) < 2048:
        raise ValueError("Audio is too short for feature extraction.")

    return audio.astype(np.float64), sr


# ==========================================
# 3. Noise augmentation
# ==========================================

def add_white_noise(audio, snr_db, rng):
    audio = np.asarray(audio, dtype=np.float64)

    signal_power = np.mean(audio ** 2)

    if signal_power < 1e-12:
        return audio.copy()

    noise = rng.normal(size=audio.shape)
    noise_power = np.mean(noise ** 2)

    desired_noise_power = signal_power / (10 ** (snr_db / 10))
    noise *= np.sqrt(desired_noise_power / (noise_power + 1e-12))

    return audio + noise


# ==========================================
# 4. Extract the same 8 DSP features
# ==========================================

FEATURE_COLUMNS = [
    "rms",
    "zcr",
    "centroid_mean",
    "centroid_std",
    "bandwidth_mean",
    "bandwidth_std",
    "rolloff_mean",
    "rolloff_std",
]


def extract_features_from_audio(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float64)

    if audio.ndim != 1 or len(audio) < 2048:
        raise ValueError("Audio must be mono and at least 2048 samples.")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite values.")

    rms = np.sqrt(np.mean(audio ** 2))

    # Fraction of adjacent sample pairs that cross zero.
    zcr = np.mean(np.abs(np.diff(np.signbit(audio))))

    frequencies, _, power = signal.spectrogram(
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
        frequencies[:, None] * power, axis=0
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
        cumulative_power >= threshold[None, :], axis=0
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


def extract_features(audio_path):
    """Convenience function for the app: path -> feature dictionary."""
    audio, sr = load_audio(audio_path)
    return extract_features_from_audio(audio, sr)


# ==========================================
# 5. Build clean + augmented feature table
# ==========================================

def main():
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {META_PATH}")

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Audio directory not found: {AUDIO_DIR}")

    meta = pd.read_csv(META_PATH)
    meta = meta[meta["category"].isin(CLASSES)].copy()

    if meta.empty:
        raise ValueError("No selected categories found in the dataset.")

    rows = []
    failures = []

    for i, row in enumerate(meta.itertuples(index=False), start=1):
        audio_path = AUDIO_DIR / row.filename

        try:
            clean_audio, sr = load_audio(audio_path)

            # Keep the clean example.
            clean_features = extract_features_from_audio(clean_audio, sr)
            rows.append({
                "filename": row.filename,
                "fold": int(row.fold),
                "category": row.category,
                "augmentation": "clean",
                **clean_features,
            })

            # Add noisy versions ONLY to folds 1-4.
            if int(row.fold) in (1, 2, 3, 4):
                for snr_db in NOISE_SNR_LEVELS:
                    rng = np.random.default_rng(
                        10000 + i * 100 + snr_db
                    )
                    noisy_audio = add_white_noise(
                        clean_audio, snr_db, rng
                    )
                    noisy_features = extract_features_from_audio(
                        noisy_audio, sr
                    )

                    rows.append({
                        "filename": f"{row.filename}__noise_{snr_db}dB",
                        "fold": int(row.fold),
                        "category": row.category,
                        "augmentation": f"noise_{snr_db}dB",
                        **noisy_features,
                    })

        except Exception as exc:
            failures.append((row.filename, str(exc)))

        if i % 20 == 0 or i == len(meta):
            print(f"Processed {i}/{len(meta)} original recordings")

    if not rows:
        raise RuntimeError("No features were extracted successfully.")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature extraction finished.")
    print("Output:", OUTPUT_PATH)
    print("Rows:", len(df))
    print("\nRows by fold:")
    print(df.groupby("fold").size())
    print("\nRows by augmentation:")
    print(df["augmentation"].value_counts())
    print("\nClasses:")
    print(df["category"].value_counts().sort_index())
    print("\nFailed original recordings:", len(failures))

    for filename, error in failures[:10]:
        print(f"FAILED: {filename}: {error}")


if __name__ == "__main__":
    main()
