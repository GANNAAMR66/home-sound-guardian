
from pathlib import Path

import pandas as pd
import soundfile as sf


# ==========================================
# 1. Find the dataset directory and metadata
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent

print("=" * 60)
print("HOME SOUND GUARDIAN - DATASET PREPARATION")
print("=" * 60)

print("\nScript directory:", SCRIPT_DIR)
print("Dataset directory:", DATASET_DIR)

# Search for esc50.csv inside the dataset directory
metadata_files = list(DATASET_DIR.rglob("esc50.csv"))

if not metadata_files:
    raise FileNotFoundError(
        "\nCannot find esc50.csv!\n"
        f"Searched inside: {DATASET_DIR}\n"
        "Make sure the ESC-50 dataset is downloaded and extracted."
    )

META_PATH = metadata_files[0]

# The standard ESC-50 structure is:
# ESC-50-master/
#     meta/esc50.csv
#     audio/*.wav
AUDIO_DIR = META_PATH.parent.parent / "audio"

print("\nMetadata path:", META_PATH)
print("Metadata exists:", META_PATH.exists())
print("Audio directory:", AUDIO_DIR)
print("Audio directory exists:", AUDIO_DIR.exists())

if not AUDIO_DIR.exists():
    raise FileNotFoundError(
        "\nCannot find the audio directory!\n"
        f"Expected location: {AUDIO_DIR}\n"
        "Check the extracted ESC-50 dataset structure."
    )


# ==========================================
# 2. Define candidate sound classes
# ==========================================

CANDIDATE_CLASSES = [
    "dog",
    "crying_baby",
    "door_wood_knock",
    "clock_alarm",
    "glass_breaking",
    "vacuum_cleaner",
    "rain",
    "thunderstorm",
]


# ==========================================
# 3. Load metadata
# ==========================================

meta = pd.read_csv(META_PATH)

required_columns = {"filename", "fold", "category"}

missing_columns = required_columns - set(meta.columns)

if missing_columns:
    raise ValueError(
        f"Metadata is missing required columns: {missing_columns}"
    )

print("\n" + "=" * 60)
print("METADATA INFORMATION")
print("=" * 60)

print("\nColumns:")
print(list(meta.columns))

print("\nFirst 5 rows:")
print(meta.head().to_string(index=False))

print("\nTotal recordings:", len(meta))


# ==========================================
# 4. Display all available categories
# ==========================================

print("\n" + "=" * 60)
print("ALL CATEGORIES AND RECORDING COUNTS")
print("=" * 60)

category_counts = meta["category"].value_counts().sort_index()
print(category_counts.to_string())


# ==========================================
# 5. Check candidate classes
# ==========================================

print("\n" + "=" * 60)
print("CANDIDATE CLASS CHECK")
print("=" * 60)

available_classes = set(meta["category"].unique())

for class_name in CANDIDATE_CLASSES:
    status = (
        "FOUND"
        if class_name in available_classes
        else "MISSING"
    )
    print(f"{class_name:20s} -> {status}")

selected_classes = [
    class_name
    for class_name in CANDIDATE_CLASSES
    if class_name in available_classes
]

if not selected_classes:
    raise ValueError(
        "None of the candidate classes were found in the metadata."
    )

subset = meta[
    meta["category"].isin(selected_classes)
].copy()

print("\nSelected classes:", selected_classes)
print("Number of selected classes:", len(selected_classes))
print("Total selected recordings:", len(subset))


# ==========================================
# 6. Check recordings per class and fold
# ==========================================

print("\n" + "=" * 60)
print("SELECTED CLASSES PER FOLD")
print("=" * 60)

fold_table = pd.crosstab(
    subset["category"],
    subset["fold"]
)

print(fold_table.to_string())

print("\nRecordings per selected class:")
print(
    subset["category"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ==========================================
# 7. Check audio files
# ==========================================

print("\n" + "=" * 60)
print("AUDIO FILE VALIDATION")
print("=" * 60)

sample_rates = set()
channel_counts = set()
durations = []
missing_files = []
unreadable_files = []

for filename in subset["filename"]:
    audio_path = AUDIO_DIR / filename

    if not audio_path.exists():
        missing_files.append(filename)
        continue

    try:
        info = sf.info(str(audio_path))

        sample_rates.add(info.samplerate)
        channel_counts.add(info.channels)
        durations.append(info.duration)

    except Exception as error:
        unreadable_files.append((filename, str(error)))


print("\nSample rates:", sorted(sample_rates))
print("Channel counts:", sorted(channel_counts))

if durations:
    print(
        f"Duration min/max: "
        f"{min(durations):.2f} / {max(durations):.2f} seconds"
    )
    print(f"Successfully checked audio files: {len(durations)}")

print("Missing audio files:", len(missing_files))
print("Unreadable audio files:", len(unreadable_files))

if missing_files:
    print("\nMissing file examples:")
    for filename in missing_files[:10]:
        print(" -", filename)

if unreadable_files:
    print("\nUnreadable file examples:")
    for filename, error in unreadable_files[:5]:
        print(f" - {filename}: {error}")


# ==========================================
# 8. Save selected metadata
# ==========================================

OUTPUT_PATH = DATASET_DIR.parent / "selected_meta.csv"

subset.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 60)
print("DATASET PREPARATION SUMMARY")
print("=" * 60)

print("Metadata source:", META_PATH)
print("Audio directory:", AUDIO_DIR)
print("Selected classes:", len(selected_classes))
print("Selected recordings:", len(subset))
print("Missing audio files:", len(missing_files))
print("Unreadable audio files:", len(unreadable_files))
print("Saved selected metadata to:", OUTPUT_PATH)

print("\nDataset inspection completed!")
