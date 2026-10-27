from pathlib import Path
from io import BytesIO
import math

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import plotly.graph_objects as go
from scipy import signal
from scipy.signal import butter, sosfiltfilt


# =========================================================
# 1. Configuration
# =========================================================
APP_DIR = Path(__file__).resolve().parent
TARGET_SAMPLE_RATE = 44100
MODEL_FILENAME = "best_sound_guardian_model.joblib"

# Support both layouts:
# 1) app.py is inside home_sound_guardian/
# 2) app.py is inside home_sound_guardian/ESC-50-master/
_PROJECT_CANDIDATES = [APP_DIR, APP_DIR / "ESC-50-master"]
PROJECT_DIR = next(
    (folder for folder in _PROJECT_CANDIDATES if (folder / "models" / MODEL_FILENAME).is_file()),
    APP_DIR,
)
MODEL_PATH = PROJECT_DIR / "models" / MODEL_FILENAME

CLASS_NAMES = {
    "dog": "Dog Barking",
    "crying_baby": "Crying Baby",
    "door_wood_knock": "Door Knocking",
    "clock_alarm": "Clock Alarm",
    "glass_breaking": "Glass Breaking",
    "vacuum_cleaner": "Vacuum Cleaner",
    "rain": "Rain",
    "thunderstorm": "Thunderstorm",
}

CLASS_ICONS = {
    "Dog Barking": "🐕",
    "Crying Baby": "👶",
    "Door Knocking": "🚪",
    "Clock Alarm": "⏰",
    "Glass Breaking": "🥂",
    "Vacuum Cleaner": "🧹",
    "Rain": "🌧️",
    "Thunderstorm": "⛈️",
}

st.set_page_config(
    page_title="Home Sound Guardian",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. Premium dark UI styling
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #071225;
        --panel: #0d1b33;
        --panel2: #102342;
        --border: rgba(105, 151, 225, .20);
        --text: #edf4ff;
        --muted: #9fb2d1;
        --blue: #438cff;
        --purple: #8a68ff;
        --teal: #21dfc0;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 0%, rgba(63, 111, 225, .16), transparent 28%),
            radial-gradient(circle at 10% 20%, rgba(113, 73, 255, .10), transparent 25%),
            #071225;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7, 18, 37, .85);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08152a 0%, #0a1730 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    h1, h2, h3, p, label, span {
        color: var(--text);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 2px 22px 2px;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 15px;
        font-size: 25px;
        background: linear-gradient(135deg, #397dff, #8c5cff);
        box-shadow: 0 8px 28px rgba(65, 115, 255, .28);
    }

    .brand-title {
        font-size: 18px;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -.4px;
    }

    .brand-subtitle {
        color: #91a8ce;
        font-size: 11px;
        margin-top: 4px;
        letter-spacing: .4px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 30px 32px;
        border: 1px solid rgba(87, 145, 245, .28);
        border-radius: 22px;
        background:
            radial-gradient(circle at 85% 50%, rgba(67, 140, 255, .25), transparent 23%),
            linear-gradient(115deg, rgba(17, 37, 72, .98), rgba(11, 26, 51, .96));
        margin-bottom: 22px;
    }

    .eyebrow {
        color: #7fb5ff;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .hero h1 {
        font-size: clamp(28px, 3vw, 42px);
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -1.3px;
        margin: 0 0 12px 0;
    }

    .gradient-text {
        background: linear-gradient(90deg, #9b83ff, #50b7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #b6c8e5;
        max-width: 680px;
        font-size: 14px;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .section-heading {
        font-size: 20px;
        font-weight: 750;
        letter-spacing: -.3px;
        margin: 8px 0 4px 0;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 14px;
    }

    .panel {
        background: linear-gradient(145deg, rgba(16, 35, 66, .95), rgba(10, 25, 49, .98));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, .12);
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 11px;
        border-radius: 999px;
        border: 1px solid rgba(33, 223, 192, .25);
        background: rgba(33, 223, 192, .08);
        color: #8bf6df;
        font-size: 11px;
        font-weight: 700;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #21dfc0;
        box-shadow: 0 0 12px #21dfc0;
    }

    .result-card {
        background: linear-gradient(135deg, rgba(10, 106, 121, .30), rgba(15, 48, 78, .72));
        border: 1px solid rgba(33, 223, 192, .28);
        border-radius: 16px;
        padding: 22px;
        min-height: 145px;
    }

    .result-label {
        color: #76f3dc;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    .result-name {
        font-size: 27px;
        font-weight: 800;
        margin: 9px 0 7px 0;
        line-height: 1.2;
    }

    .result-note {
        color: #a9c9db;
        font-size: 12px;
    }

    .mini-card {
        background: rgba(14, 32, 60, .88);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 15px 16px;
        min-height: 94px;
    }

    .mini-label {
        color: #9eb4d7;
        font-size: 11px;
        margin-bottom: 8px;
    }

    .mini-value {
        color: #f2f6ff;
        font-size: 20px;
        font-weight: 750;
    }

    .feature-card {
        background: rgba(12, 29, 55, .85);
        border: 1px solid rgba(105, 151, 225, .16);
        border-radius: 13px;
        padding: 14px 10px;
        text-align: center;
        min-height: 93px;
        margin-bottom: 8px;
    }

    .feature-name {
        color: #a8bbdb;
        font-size: 10px;
        min-height: 28px;
    }

    .feature-value {
        color: #f1f6ff;
        font-size: 15px;
        font-weight: 750;
        overflow-wrap: anywhere;
    }

    .footer {
        text-align: center;
        color: #7188ad;
        font-size: 11px;
        padding: 24px 0 8px 0;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #5145ff, #168cf7);
        color: white;
        border: 0;
        border-radius: 12px;
        font-weight: 700;
        min-height: 48px;
        box-shadow: 0 7px 20px rgba(60, 106, 255, .20);
    }

    div.stButton > button[kind="primary"]:hover {
        border: 1px solid rgba(255,255,255,.3);
        color: white;
        filter: brightness(1.08);
    }

    [data-testid="stFileUploader"] {
        background: rgba(12, 28, 54, .65);
        border: 1px dashed rgba(90, 151, 255, .42);
        border-radius: 14px;
        padding: 10px;
    }

    [data-testid="stMetric"] {
        background: rgba(14, 32, 60, .82);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #9eb4d7 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f2f6ff !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    .stAlert {
        border-radius: 12px;
    }

    @media (max-width: 800px) {
        .hero { padding: 22px 20px; }
        .result-name { font-size: 22px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. Load model
# =========================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        expected = [str(folder / "models" / MODEL_FILENAME) for folder in _PROJECT_CANDIDATES]
        raise FileNotFoundError(
            "Trained model was not found. Checked these locations:\n- " + "\n- ".join(expected)
        )

    saved = joblib.load(MODEL_PATH)
    if not isinstance(saved, dict) or "model" not in saved or "feature_columns" not in saved:
        raise ValueError(
            "The model file does not contain the expected bundle keys: 'model' and 'feature_columns'. "
            "Run train_model.py again to create the current model bundle."
        )

    model = saved["model"]
    feature_columns = list(saved["feature_columns"])
    expected_features = [
        "rms", "zcr", "centroid_mean", "centroid_std",
        "bandwidth_mean", "bandwidth_std", "rolloff_mean", "rolloff_std",
    ]
    missing = [name for name in expected_features if name not in feature_columns]
    if missing:
        raise ValueError(f"The loaded model is missing expected features: {missing}")

    return model, feature_columns, saved.get("model_name", type(model).__name__)


# =========================================================
# 4. Audio preprocessing, DSP tools and features
# =========================================================
def preprocess_audio(uploaded_file):
    audio, sample_rate = sf.read(
        BytesIO(uploaded_file.getvalue()),
        dtype="float64",
    )

    if audio.size == 0:
        raise ValueError("The uploaded audio file is empty.")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if not np.isfinite(audio).all():
        raise ValueError("The audio contains invalid values.")

    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), TARGET_SAMPLE_RATE)
        up = TARGET_SAMPLE_RATE // divisor
        down = int(sample_rate) // divisor
        audio = signal.resample_poly(audio, up, down)
        sample_rate = TARGET_SAMPLE_RATE

    if len(audio) < 2048:
        raise ValueError("Audio is too short for feature extraction.")

    return audio, sample_rate


def add_noise(audio, snr_db, seed=42):
    """Add white Gaussian noise so that the result has the requested SNR (dB)."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(audio ** 2)

    if signal_power < 1e-12:
        return audio.copy()

    noise = rng.normal(size=audio.shape)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise *= np.sqrt(target_noise_power / np.mean(noise ** 2))

    return audio + noise


def apply_bandpass(audio, sample_rate):
    """Butterworth band-pass (100 Hz to 8 kHz, order 4), zero-phase filtering."""
    low_cutoff = 100.0
    high_cutoff = min(8000.0, sample_rate * 0.45)

    sos = butter(
        4,
        [low_cutoff, high_cutoff],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )

    return sosfiltfilt(sos, audio)


def extract_features(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float64)
    rms = np.sqrt(np.mean(audio ** 2))
    zcr = np.mean(np.abs(np.diff(np.signbit(audio))))

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
    centroid = np.sum(frequencies[:, None] * power, axis=0) / power_sum

    bandwidth = np.sqrt(
        np.sum(((frequencies[:, None] - centroid) ** 2) * power, axis=0)
        / power_sum
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


# =========================================================
# 5. Visualization helpers
# =========================================================
def chart_layout(fig, height=270):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d8f2", family="Inter, sans-serif", size=11),
        xaxis=dict(
            gridcolor="rgba(133,164,211,.10)",
            zerolinecolor="rgba(133,164,211,.14)",
        ),
        yaxis=dict(
            gridcolor="rgba(133,164,211,.10)",
            zerolinecolor="rgba(133,164,211,.14)",
        ),
        showlegend=False,
    )
    return fig


def make_waveform(audio, sample_rate):
    # Downsample the display points only; analysis still uses the full signal.
    max_points = 5000
    step = max(1, len(audio) // max_points)
    display_audio = audio[::step]
    times = np.arange(len(display_audio)) * step / sample_rate

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=display_audio,
            mode="lines",
            line=dict(color="#36a3ff", width=1.15),
            fill="tozeroy",
            fillcolor="rgba(54,163,255,.10)",
            hovertemplate="Time: %{x:.3f}s<br>Amplitude: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Audio waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
    )
    return chart_layout(fig, 260)


def make_spectrogram(audio, sample_rate):
    frequencies, times, power = signal.spectrogram(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=2048,
        noverlap=1024,
        detrend=False,
        scaling="spectrum",
        mode="psd",
    )
    power_db = 10 * np.log10(power + 1e-12)
    fig = go.Figure(
        data=go.Heatmap(
            x=times,
            y=frequencies,
            z=power_db,
            colorscale=[
                [0.0, "#101b40"],
                [0.25, "#3430a0"],
                [0.5, "#236ee8"],
                [0.75, "#25c5c5"],
                [1.0, "#f4df6a"],
            ],
            colorbar=dict(
                title="dB",
                thickness=10,
                tickfont=dict(color="#c9d8f2"),
            ),
            hovertemplate="Time: %{x:.2f}s<br>Frequency: %{y:.0f}Hz<br>Power: %{z:.1f}dB<extra></extra>",
        )
    )
    fig.update_layout(
        title="Spectrogram",
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
    )
    return chart_layout(fig, 300)


def render_feature_cards(feature_values):
    pretty_names = {
        "rms": "RMS",
        "zcr": "Zero Crossing Rate",
        "centroid_mean": "Centroid Mean",
        "centroid_std": "Centroid Std",
        "bandwidth_mean": "Bandwidth Mean",
        "bandwidth_std": "Bandwidth Std",
        "rolloff_mean": "Rolloff Mean",
        "rolloff_std": "Rolloff Std",
    }

    cols = st.columns(4)
    for index, (key, value) in enumerate(feature_values.items()):
        with cols[index % 4]:
            if key in ("rms", "zcr"):
                formatted = f"{value:.4f}"
            else:
                formatted = f"{value:,.1f}"
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-name">{pretty_names.get(key, key)}</div>
                    <div class="feature-value">{formatted}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# 6. Load model safely (must happen BEFORE the sidebar,
#    because the sidebar shows the loaded model's name)
# =========================================================
try:
    model, feature_columns, model_name = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# =========================================================
# 7. Sidebar
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">〽</div>
            <div>
                <div class="brand-title">Home Sound<br>Guardian</div>
                <div class="brand-subtitle">LISTEN · DETECT · PROTECT</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Model Performance", "About Project"],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div class="status-pill"><span class="status-dot"></span> AI MODEL READY</div>
        <br><br>
        <div style="color:#91a8ce;font-size:12px;line-height:1.8">
            <b style="color:#edf4ff">Current model</b><br>
            {model_name} classifier<br><br>
            <b style="color:#edf4ff">Audio format</b><br>
            WAV · Mono processing<br><br>
            <b style="color:#edf4ff">Target sample rate</b><br>
            {TARGET_SAMPLE_RATE:,} Hz<br><br>
            <b style="color:#edf4ff">Sound categories</b><br>
            8 supported classes
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("DSP • Feature Extraction • Machine Learning")


# =========================================================
# 8. Dashboard page
# =========================================================
if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">SMART AUDIO ANALYTICS</div>
            <h1>Home Sound <span class="gradient-text">Guardian</span></h1>
            <p>
                Understand the sounds around you. Upload a WAV recording and let
                digital signal processing and machine learning classify the
                sound into one of eight supported categories.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    upload_col, tips_col = st.columns([1.12, 0.88], gap="large")

    with upload_col:
        st.markdown('<div class="section-heading">Analyze a sound</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Upload a WAV file to begin your analysis</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav"],
            help="Use a WAV file. Audio is converted to mono and resampled to 44.1 kHz if needed.",
            label_visibility="visible",
        )
        if uploaded_file is not None:
            st.audio(uploaded_file.getvalue(), format="audio/wav")
            st.caption(f"Selected file: {uploaded_file.name}")

        # ---- Optional DSP pre-processing (applied BEFORE classification) ----
        st.markdown("**DSP pre-processing (optional)**")
        noise_on = st.checkbox(
            "Add white noise",
            help="Simulates a noisy environment by adding white Gaussian noise.",
        )
        snr_db = st.slider(
            "SNR (dB)  ·  lower = noisier",
            min_value=0,
            max_value=30,
            value=10,
            disabled=not noise_on,
        )
        filter_on = st.checkbox(
            "Apply band-pass filter (100–8000 Hz)",
            help="Butterworth filter, order 4, applied forward and backward (zero delay).",
        )

        analyze_clicked = st.button(
            "✦  Analyze Sound",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )
        st.markdown(
            '<div style="color:#91a8ce;font-size:11px;margin-top:10px">Supported input: WAV · Recommended sample rate: 44.1 kHz</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tips_col:
        st.markdown('<div class="section-heading">Supported sounds</div>', unsafe_allow_html=True)
        st.markdown(
                   '<div class="section-subtitle">Model selection results on the validation split (clean recordings held out from folds 1-4)</div>',
            unsafe_allow_html=True,
        )
        category_cols = st.columns(2)
        for i, display_name in enumerate(CLASS_NAMES.values()):
            with category_cols[i % 2]:
                st.markdown(
                    f"""
                    <div class="mini-card" style="margin-bottom:10px;min-height:75px">
                        <div style="font-size:22px;margin-bottom:5px">{CLASS_ICONS.get(display_name, "🔊")}</div>
                        <div style="font-size:12px;font-weight:650">{display_name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if analyze_clicked and uploaded_file is not None:
        try:
            with st.spinner("Processing audio and extracting DSP features..."):
                audio, sample_rate = preprocess_audio(uploaded_file)

                # Optional DSP steps: noise first, then filter.
                if noise_on:
                    audio = add_noise(audio, snr_db)
                if filter_on:
                    audio = apply_bandpass(audio, sample_rate)

                # Keep a playable copy of exactly what the classifier receives.
                processed_wav = BytesIO()
                sf.write(processed_wav, np.clip(audio, -1.0, 1.0), sample_rate, format="WAV")

                duration = len(audio) / sample_rate
                feature_values = extract_features(audio, sample_rate)

                missing_features = [column for column in feature_columns if column not in feature_values]
                if missing_features:
                    raise ValueError(
                        "The app did not calculate all features expected by the model: "
                        f"{missing_features}"
                    )

                # Important: preserve the exact feature order stored with the trained model.
                X_one = pd.DataFrame(
                    [[feature_values[column] for column in feature_columns]],
                    columns=feature_columns,
                )
                if not np.isfinite(X_one.to_numpy(dtype=float)).all():
                    raise ValueError("Feature extraction produced NaN or infinite values.")

                prediction = model.predict(X_one)[0]
                predicted_name = CLASS_NAMES.get(prediction, str(prediction))
                confidence = None
                probability_df = None

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_one)[0]
                    class_labels = model.classes_
                    probability_df = pd.DataFrame({
                        "Sound Category": [
                            CLASS_NAMES.get(label, str(label))
                            for label in class_labels
                        ],
                        "Model Score": probabilities,
                    }).sort_values("Model Score", ascending=False)
                    best_index = int(np.argmax(probabilities))
                    confidence = float(probabilities[best_index])

            st.markdown('<div class="section-heading">Analysis results</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">Classification output and signal insights for your recording</div>',
                unsafe_allow_html=True,
            )

            # Tell the user which DSP steps were applied.
            applied_steps = []
            if noise_on:
                applied_steps.append(f"white noise at {snr_db} dB SNR")
            if filter_on:
                applied_steps.append("band-pass filter (100–8000 Hz)")
            if applied_steps:
                st.info("DSP pre-processing applied before classification: " + " + ".join(applied_steps))

            result_col, stats_col = st.columns([1.05, 1], gap="large")
            with result_col:
                score_text = f"{confidence:.1%} model score" if confidence is not None else "Prediction generated"
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Predicted category</div>
                        <div style="font-size:34px;margin-top:9px">{CLASS_ICONS.get(predicted_name, "🔊")}</div>
                        <div class="result-name">{predicted_name}</div>
                        <div class="result-note">{score_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("")
                if confidence is not None:
                    st.progress(confidence, text="Top class model score")
                    st.caption("Model scores are not a guarantee of correctness and may be overconfident.")

            with stats_col:
                stat1, stat2 = st.columns(2)
                with stat1:
                    st.metric("Duration", f"{duration:.2f} s")
                with stat2:
                    st.metric("Sample rate", f"{sample_rate / 1000:.1f} kHz")
                stat3, stat4 = st.columns(2)
                with stat3:
                    st.metric("Audio channels", "Mono")
                with stat4:
                    st.metric("DSP features", str(len(feature_values)))

            if noise_on or filter_on:
                st.markdown("**Audio actually sent to the classifier**")
                st.audio(processed_wav.getvalue(), format="audio/wav")

            if probability_df is not None:
                left, right = st.columns([1, 1], gap="large")
                with left:
                    st.markdown('<div class="section-heading">Top predictions</div>', unsafe_allow_html=True)
                    top_three = probability_df.head(3).copy()
                    for _, row in top_three.iterrows():
                        st.markdown(
                            f"**{row['Sound Category']}**  ·  {row['Model Score']:.1%}"
                        )
                        st.progress(float(row["Model Score"]))
                with right:
                    st.markdown('<div class="section-heading">Class score distribution</div>', unsafe_allow_html=True)
                    bar_fig = go.Figure(
                        go.Bar(
                            x=probability_df["Model Score"],
                            y=probability_df["Sound Category"],
                            orientation="h",
                            marker=dict(
                                color=probability_df["Model Score"],
                                colorscale=[[0, "#3b4c80"], [0.55, "#477dff"], [1, "#28ddc0"]],
                            ),
                            hovertemplate="%{y}: %{x:.1%}<extra></extra>",
                        )
                    )
                    bar_fig.update_layout(
                        xaxis=dict(tickformat=".0%", range=[0, 1], title="Model score"),
                        yaxis=dict(autorange="reversed"),
                        title="Scores for all supported classes",
                    )
                    st.plotly_chart(chart_layout(bar_fig, 320), use_container_width=True)

            st.markdown('<div class="section-heading">Signal visualization</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">Waveform shows amplitude over time; spectrogram shows how frequency energy changes over time</div>',
                unsafe_allow_html=True,
            )
            wave_col, spec_col = st.columns(2, gap="large")
            with wave_col:
                st.plotly_chart(make_waveform(audio, sample_rate), use_container_width=True)
            with spec_col:
                st.plotly_chart(make_spectrogram(audio, sample_rate), use_container_width=True)

            st.markdown('<div class="section-heading">Extracted DSP features</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-subtitle">The numerical features supplied to the trained classifier</div>',
                unsafe_allow_html=True,
            )
            render_feature_cards(feature_values)
            feature_df = pd.DataFrame({
                "Feature": list(feature_values.keys()),
                "Value": list(feature_values.values()),
            })
            st.dataframe(feature_df, use_container_width=True, hide_index=True)

            report = feature_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download extracted features (CSV)",
                data=report,
                file_name="home_sound_guardian_features.csv",
                mime="text/csv",
            )

            st.info(
                "Research note: this model always selects a known class. "
                "Sounds outside the eight training categories may be misclassified. "
                "The score is a model estimate, not a calibrated measure of certainty."
            )

        except Exception as error:
            st.error(f"Audio analysis failed: {error}")

    elif uploaded_file is None:
        st.markdown(
            """
            <div class="panel">
                <div style="font-size:16px;font-weight:750;margin-bottom:7px">Ready when you are</div>
                <div style="color:#9fb2d1;font-size:13px;line-height:1.7">
                    Choose a WAV recording above. The app will preprocess the audio,
                    calculate eight DSP features, and predict its sound category.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# 9. Model performance page
# =========================================================
elif page == "Model Performance":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">EVALUATION CENTER</div>
            <h1>Model <span class="gradient-text">Performance</span></h1>
            <p>Review the held-out test results from the experiments run for this project.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Model comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Results recorded from the fold-5 test set (64 recordings)</div>',
        unsafe_allow_html=True,
    )

    comparison_path = PROJECT_DIR / "results" / "model_comparison.csv"
    # Support alternate naming if the project uses a different output filename.
    if not comparison_path.exists():
        candidates = list((PROJECT_DIR / "results").glob("*comparison*.csv")) if (PROJECT_DIR / "results").exists() else []
        candidates = [p for p in candidates if "robustness" not in p.name.lower()]
        if candidates:
            comparison_path = candidates[0]

    if comparison_path.exists():
        try:
            comparison_df = pd.read_csv(comparison_path)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download model comparison CSV",
                data=comparison_df.to_csv(index=False).encode("utf-8"),
                file_name="model_comparison.csv",
                mime="text/csv",
            )
        except Exception as error:
            st.warning(f"Could not read comparison results: {error}")
    else:
        st.warning(
            "Model comparison CSV was not found in the results folder. "
            "The dashboard still works; place the generated comparison CSV inside results/ to display it here."
        )

    st.markdown('<div class="section-heading">DSP robustness test</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Check how background noise and filtering affected predictions</div>',
        unsafe_allow_html=True,
    )
    robustness_path = PROJECT_DIR / "results" / "dsp_robustness_comparison.csv"
    if robustness_path.exists():
        try:
            robustness_df = pd.read_csv(robustness_path)
            st.dataframe(robustness_df, use_container_width=True, hide_index=True)
        except Exception as error:
            st.warning(f"Could not read robustness results: {error}")
    else:
        st.info("Robustness results will appear here when results/dsp_robustness_comparison.csv is available.")

    st.markdown(
        """
        <div class="panel">
            <b>Interpretation</b>
            <p style="color:#a8bbdb;font-size:13px;line-height:1.7;margin-top:8px">
                Accuracy measures the proportion of correct predictions. Macro F1 averages
                the F1 score across classes, giving each class equal weight. Robustness
                results help show whether the classifier remains useful when audio is noisy
                or filtered.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 10. About page
# =========================================================
else:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">ABOUT THE PROJECT</div>
            <h1>Built with DSP <span class="gradient-text">& Machine Learning</span></h1>
            <p>A sound classification prototype created to connect signal processing concepts with a practical AI interface.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(
            """
            <div class="panel">
                <div class="section-heading">Processing pipeline</div>
                <p style="color:#a8bbdb;font-size:13px;line-height:1.9">
                    <b>01</b> · Upload WAV audio<br>
                    <b>02</b> · Convert to mono and resample to 44.1 kHz<br>
                    <b>03</b> · (Optional) add noise and/or band-pass filter<br>
                    <b>04</b> · Extract eight DSP features<br>
                    <b>05</b> · Predict the sound category<br>
                    <b>06</b> · Visualize the signal and model scores
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
            <div class="panel">
                <div class="section-heading">DSP features</div>
                <p style="color:#a8bbdb;font-size:13px;line-height:1.9">
                    <b>RMS</b> · Signal energy level<br>
                    <b>ZCR</b> · Sign-change rate in the signal<br>
                    <b>Spectral centroid</b> · Spectral brightness estimate<br>
                    <b>Bandwidth</b> · Spread of spectral energy<br>
                    <b>Rolloff</b> · Frequency below which 85% of power lies
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-heading">Supported categories</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for index, name in enumerate(CLASS_NAMES.values()):
        with cols[index % 4]:
            st.markdown(
                f"""
                <div class="mini-card" style="margin-bottom:12px">
                    <div style="font-size:25px;margin-bottom:7px">{CLASS_ICONS.get(name, "🔊")}</div>
                    <div style="font-size:12px;font-weight:700">{name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(f"Loaded model: {model_name} · Input sample rate: {TARGET_SAMPLE_RATE:,} Hz")
    st.caption("Dataset: ESC-50 (CC BY-NC 4.0) · Piczak, K. J., 2015")


st.markdown(
    '<div class="footer">HOME SOUND GUARDIAN · DSP + MACHINE LEARNING · PROJECT DEMONSTRATION</div>',
    unsafe_allow_html=True,
)