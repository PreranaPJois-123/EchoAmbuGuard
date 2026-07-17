"""
app.py
-------
EchoAmbuGuard — main Streamlit entry point. Provides navigation between:
  1. Record / Upload Audio
  2. Train Model (training_ui.py)
  3. Predict Ambulance / Non-Ambulance
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.utils import load_config
from src.prediction import load_trained_model, predict_file
from training_ui import render_training_page

st.set_page_config(page_title="EchoAmbuGuard", page_icon="🚑", layout="wide")

cfg = load_config()

st.sidebar.title("🚑 EchoAmbuGuard")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🎙️ Record / Upload Audio", "🧠 Train Model", "🔮 Predict"],
)


def render_home():
    st.header("EchoAmbuGuard — Ambulance Siren Detection")
    st.write(
        "A CNN-based ambulance siren detector built on MFCC audio features. "
        "Use the sidebar to upload training audio, train the model, and run "
        "predictions on new recordings."
    )
    st.markdown(
        """
        **Pipeline**
        1. Record or Upload Audio → `dataset/<class_name>/`
        2. Extract MFCC Features → `features/`
        3. Train CNN Model → `models/`
        4. Predict Ambulance / Non-Ambulance
        """
    )


def render_upload_page():
    st.header("🎙️ Record / Upload Audio")
    class_name = st.selectbox("Which class does this audio belong to?", cfg.classes)
    dest_dir = cfg.dataset_path / class_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    uploaded = st.file_uploader(
        "Upload one or more audio files (.wav, .mp3, .flac, .ogg, .m4a)",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        accept_multiple_files=True,
    )
    if uploaded:
        saved = 0
        for f in uploaded:
            out_path = dest_dir / f.name
            with open(out_path, "wb") as out:
                out.write(f.getbuffer())
            saved += 1
        st.success(f"Saved {saved} file(s) to `{dest_dir}`.")

    recorded = st.audio_input("Or record directly from your microphone")
    if recorded is not None:
        out_path = dest_dir / f"recorded_{class_name}_{len(list(dest_dir.glob('*')))}.wav"
        with open(out_path, "wb") as out:
            out.write(recorded.getbuffer())
        st.success(f"Recording saved to `{out_path}`.")

    st.subheader("Current dataset contents")
    for cls in cfg.classes:
        files = list((cfg.dataset_path / cls).glob("*"))
        files = [f for f in files if f.name != ".gitkeep"]
        st.write(f"**{cls}**: {len(files)} file(s)")


def render_predict_page():
    st.header("🔮 Predict Ambulance / Non-Ambulance")
    model, metadata = load_trained_model(cfg)

    if model is None:
        st.warning("No trained model found yet. Go to '🧠 Train Model' first.")
        return

    if metadata:
        with st.expander("Model metadata"):
            st.json(metadata)

    audio_file = st.file_uploader(
        "Upload an audio clip to classify",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        key="predict_upload",
    )
    recorded = st.audio_input("Or record audio to classify", key="predict_record")

    file_to_predict: Path | None = None
    tmp_dir = cfg.logs_path / "tmp_predictions"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if audio_file is not None:
        file_to_predict = tmp_dir / audio_file.name
        with open(file_to_predict, "wb") as out:
            out.write(audio_file.getbuffer())
    elif recorded is not None:
        file_to_predict = tmp_dir / "recorded_predict.wav"
        with open(file_to_predict, "wb") as out:
            out.write(recorded.getbuffer())

    if file_to_predict is not None:
        st.audio(str(file_to_predict))
        if st.button("Predict", type="primary"):
            try:
                with st.spinner("Running inference..."):
                    result = predict_file(file_to_predict, cfg, model, metadata)
                label = result["label"]
                confidence = result["confidence"]

                if label == "ambulance":
                    st.error(f"🚨 **AMBULANCE DETECTED** (confidence: {confidence:.1%})")
                else:
                    st.success(f"✅ Non-ambulance (confidence: {confidence:.1%})")

                st.bar_chart(result["probabilities"])
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)


if page == "🏠 Home":
    render_home()
elif page == "🎙️ Record / Upload Audio":
    render_upload_page()
elif page == "🧠 Train Model":
    render_training_page()
elif page == "🔮 Predict":
    render_predict_page()
