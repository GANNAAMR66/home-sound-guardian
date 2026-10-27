
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)


# ==========================================
# 1. Paths
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

FEATURES_PATH = PROJECT_DIR / "features.csv"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_sound_guardian_model.joblib"
COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
REPORT_PATH = RESULTS_DIR / "model_comparison_report.txt"

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

TARGET = "category"
RANDOM_STATE = 42


# ==========================================
# 2. Load and validate features
# ==========================================

if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Missing {FEATURES_PATH}. Run feature_extraction.py first."
    )

df = pd.read_csv(FEATURES_PATH)

required = FEATURE_COLUMNS + [
    TARGET, "fold", "filename", "augmentation"
]
missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing columns in features.csv: {missing}")

if df[required].isnull().any().any():
    raise ValueError("Missing values found in features.csv.")

if not np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
    raise ValueError("Non-finite feature values found.")

df["fold"] = df["fold"].astype(int)

# Official split: folds 1-4 train/validation; fold 5 test.
development_df = df[df["fold"].isin([1, 2, 3, 4])].copy()
test_df = df[df["fold"] == 5].copy()

if development_df.empty or test_df.empty:
    raise ValueError("Development or Fold 5 test set is empty.")

# The validation split is grouped by ORIGINAL recording.
# No noisy version of a validation recording may enter training.
development_df["source_id"] = (
    development_df["filename"]
    .str.replace(r"__noise_(20|10)dB$", "", regex=True)
)

clean_development = development_df[
    development_df["augmentation"] == "clean"
].copy()


# Build a normal NumPy array to avoid Arrow indexing issues.
source_table = (
    clean_development[["source_id", TARGET]]
    .drop_duplicates(subset="source_id")
    .reset_index(drop=True)
)

source_ids = source_table["source_id"].astype(str).to_numpy()
source_labels = source_table[TARGET].astype(str).to_numpy()

train_sources, val_sources = train_test_split(
    source_ids,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=source_labels,
)

train_sources = set(train_sources.tolist())
val_sources = set(val_sources.tolist())


train_sources = set(train_sources)
val_sources = set(val_sources)

train_df = development_df[
    development_df["source_id"].isin(train_sources)
].copy()

val_df = development_df[
    development_df["source_id"].isin(val_sources)
    & (development_df["augmentation"] == "clean")
].copy()

# Test only original clean Fold 5 recordings.
test_clean_df = test_df[
    test_df["augmentation"] == "clean"
].copy()

# Ensure all classes exist in each split.
classes = sorted(df[TARGET].unique())

for split_name, split_df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_clean_df),
]:
    absent = set(classes) - set(split_df[TARGET].unique())
    if absent:
        raise ValueError(
            f"Classes missing from {split_name}: {sorted(absent)}"
        )

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET]

X_val = val_df[FEATURE_COLUMNS]
y_val = val_df[TARGET]

X_test = test_clean_df[FEATURE_COLUMNS]
y_test = test_clean_df[TARGET]

print("=" * 65)
print("HOME SOUND GUARDIAN - TRAINING")
print("=" * 65)
print("Training rows including augmentation:", len(train_df))
print("Validation clean recordings:", len(val_df))
print("Fold 5 clean test recordings:", len(test_clean_df))
print("Features:", len(FEATURE_COLUMNS))
print("Classes:", classes)
print("\nTraining augmentation counts:")
print(train_df["augmentation"].value_counts())


# ==========================================
# 3. Models
# ==========================================

models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
        ),
    ),
}


# ==========================================
# 4. Train and select using validation only
# ==========================================

results = []
trained_models = {}
reports = {}

for name, model in models.items():
    print("\n" + "-" * 65)
    print("Training:", name)

    model.fit(X_train, y_train)

    # Training score is diagnostic only; it is not the selection score.
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(
        y_val, val_pred, average="macro", zero_division=0
    )

    results.append({
        "model": name,
        "train_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "validation_macro_f1": val_f1,
    })

    trained_models[name] = model
    reports[name] = classification_report(
        y_val,
        val_pred,
        labels=classes,
        target_names=classes,
        zero_division=0,
    )

    print(f"Train accuracy:      {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation Macro F1: {val_f1:.4f}")
    print("\nValidation report:")
    print(reports[name])

comparison = pd.DataFrame(results).sort_values(
    by="validation_macro_f1",
    ascending=False,
).reset_index(drop=True)

comparison.to_csv(COMPARISON_PATH, index=False)

# Select by validation Macro F1, not by Fold 5.
best_name = comparison.iloc[0]["model"]
best_model = trained_models[best_name]

# Evaluate the selected model on Fold 5 once.
test_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(
    y_test, test_pred, average="macro", zero_division=0
)

test_report = classification_report(
    y_test,
    test_pred,
    labels=classes,
    target_names=classes,
    zero_division=0,
)

print("\n" + "=" * 65)
print("SELECTED MODEL:", best_name)
print("Validation Macro F1:", f"{comparison.iloc[0]['validation_macro_f1']:.4f}")
print("Fold 5 test accuracy:", f"{test_acc:.4f}")
print("Fold 5 test Macro F1:", f"{test_f1:.4f}")
print("\nFold 5 test report:")
print(test_report)

# Confusion matrix for Fold 5.
fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    test_pred,
    labels=classes,
    display_labels=classes,
    xticks_rotation=45,
    values_format="d",
    ax=ax,
    colorbar=False,
)
ax.set_title(f"Fold 5 Confusion Matrix - {best_name}")
fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "best_model_fold5_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write("MODEL COMPARISON\n")
    file.write(comparison.to_string(index=False))
    file.write("\n\nSELECTED MODEL: " + best_name + "\n")
    file.write("\nFOLD 5 TEST REPORT\n")
    file.write(test_report)

# Save metadata with the model.
joblib.dump(
    {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "classes": classes,
        "target_sample_rate": 44100,
    },
    MODEL_PATH,
)

print("\nComparison saved:", COMPARISON_PATH)
print("Report saved:", REPORT_PATH)
print("Model saved:", MODEL_PATH)
print("\nTraining completed.")
