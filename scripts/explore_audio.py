
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal


# ==========================================
# 1. Locate the dataset
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent / "data" / "ESC-50-master"

META_PATH = DATASET_ROOT / "meta" / "esc50.csv"
AUDIO_DIR = DATASET_ROOT / "audio"

if not META_PATH.exists():
    raise FileNotFoundError(f"Metadata not found: {META_PATH}")

meta = pd.read_csv(META_PATH)

# Select one dog recording for our first experiment
dog_rows = meta[meta["category"] == "dog"]

if dog_rows.empty:
    raise ValueError("No dog recordings found in the metadata.")

filename = dog_rows.iloc[0]["filename"]
audio_path = AUDIO_DIR / filename

if not audio_path.exists():
    raise FileNotFoundError(f"Audio file not found: {audio_path}")


# ==========================================
# 2. Load the audio signal
# ==========================================

audio, sample_rate = sf.read(audio_path)

# ESC-50 recordings are expected to be mono.
if audio.ndim > 1:
    audio = audio.mean(axis=1)

duration = len(audio) / sample_rate
time = np.arange(len(audio)) / sample_rate

print("Filename:", filename)
print("Category: dog")
print("Sample rate:", sample_rate, "Hz")
print("Number of samples:", len(audio))
print("Duration:", round(duration, 2), "seconds")
print("Minimum amplitude:", round(float(audio.min()), 4))
print("Maximum amplitude:", round(float(audio.max()), 4))


# ==========================================
# 3. Calculate basic DSP features
# ==========================================

# RMS: signal energy / loudness-related measure
rms = np.sqrt(np.mean(audio ** 2))

# ZCR: fraction of neighboring samples that change sign
zcr = np.mean(
    np.abs(np.diff(np.signbit(audio)))
)

# Spectral centroid
frequencies, times, spectrum = signal.spectrogram(
    audio,
    fs=sample_rate,
    nperseg=2048,
    noverlap=1024
)

magnitude = np.sqrt(spectrum)

centroid_per_frame = (
    np.sum(frequencies[:, None] * magnitude, axis=0)
    / (np.sum(magnitude, axis=0) + 1e-10)
)

spectral_centroid = float(np.mean(centroid_per_frame))

print("\nDSP Features")
print("RMS:", round(float(rms), 5))
print("Zero-Crossing Rate:", round(float(zcr), 5))
print(
    "Spectral Centroid:",
    round(spectral_centroid, 2),
    "Hz"
)


# ==========================================
# 4. Plot the waveform
# ==========================================

plt.figure(figsize=(12, 4))
plt.plot(time, audio)
plt.title(f"Waveform - {filename}")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()


# ==========================================
# 5. Plot the spectrogram
# ==========================================

plt.figure(figsize=(12, 5))

plt.pcolormesh(
    times,
    frequencies,
    10 * np.log10(spectrum + 1e-12),
    shading="auto"
)

plt.title(f"Spectrogram - {filename}")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Power (dB)")
plt.tight_layout()
plt.show()
