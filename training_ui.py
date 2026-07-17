"""
training_ui.py
----------------
The Streamlit "Train Model" page.

IMPORTANT DESIGN NOTE (this is the fix for your bug):
model.fit() is invoked directly inside this button handler, on Streamlit's
main script-execution thread. It is NEVER wrapped in threading.Thread or
multiprocessing. Streamlit widgets (st.progress, st.empty, st.metric) are
only ever touched from that same main thread, inside the ProgressCallback
hook — which is safe and renders correctly, unlike calling st.* from a
background thread (which silently fails and looks like a permanent hang).

model.fit() itself runs with verbose=2 (clean flushed per-epoch log lines,
no carriage-return progress bar that certain terminals swallow) AND with an
epoch-end callback that updates the on-page progress bar/metrics live.
"""

from __future__ import annotations

import time

import streamlit as st

from src.utils import load_config, get_logger
from src.dataset_manager import dataset_statistics
from src.train_engine import run_training, TrainingProgress
from src.plotting import plot_training_curves, plot_confusion_matrix
from src.feature_extractor import extract_and_save_all, features_exist


def render_training_page():
    cfg = load_config()
    logger = get_logger("training_ui", cfg)

    st.header("🚑 Train EchoAmbuGuard CNN")

    # ----------------------------------------------------------------- #
    # 1. Feature extraction trigger (if not already done)
    # ----------------------------------------------------------------- #
    with st.expander("Step 1 — Feature Extraction", expanded=not features_exist(cfg)):
        st.write("Extract MFCC features from `dataset/<class_name>/*` before training.")
        if st.button("🔍 Extract Features Now", key="extract_btn"):
            with st.spinner("Extracting MFCC features..."):
                summary = extract_and_save_all(cfg)
            if summary["total_saved"] == 0:
                st.error("No valid features were extracted. Check your audio files.")
            else:
                st.success(f"Extracted {summary['total_saved']} feature vectors "
                           f"({summary['classes']}).")
                if summary["skipped"]:
                    st.warning(f"Skipped {len(summary['skipped'])} corrupted/unreadable "
                               f"files: {summary['skipped']}")

    # ----------------------------------------------------------------- #
    # 2. Dataset statistics (requirement: show BEFORE training)
    # ----------------------------------------------------------------- #
    st.subheader("📊 Dataset Statistics")
    stats = dataset_statistics(cfg)

    if not stats["available"]:
        st.info("No extracted features found yet. Run Step 1 above first.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", stats["total_samples"])
    col2.metric("Feature Shape", str(stats["feature_shape"]))
    col3.metric("Classes", len(cfg.classes))

    st.bar_chart(stats["class_counts"])
    st.caption(f"Class distribution: {stats['class_counts']}  |  dtype: {stats['dtype']}")

    for cls, count in stats["class_counts"].items():
        if count < 5:
            st.warning(f"⚠️ Class '{cls}' has only {count} samples. With very small "
                       f"datasets, validate results carefully and consider adding more data.")

    # ----------------------------------------------------------------- #
    # 3. Training configuration
    # ----------------------------------------------------------------- #
    st.subheader("⚙️ Training Configuration")
    c1, c2 = st.columns(2)
    epochs = c1.number_input("Epochs", min_value=1, max_value=500, value=cfg.epochs, step=1)
    batch_size = c2.number_input("Batch Size", min_value=1, max_value=cfg.max_batch_size,
                                  value=min(cfg.batch_size, stats["total_samples"]), step=1)

    est_seconds = max(2.0, stats["total_samples"] / max(batch_size, 1)) * epochs * 0.15
    st.caption(f"⏱️ Estimated training time: ~{est_seconds:.0f} seconds "
               f"(rough estimate for a CPU on this dataset size).")

    # ----------------------------------------------------------------- #
    # 4. Train button — SYNCHRONOUS call, no threads
    # ----------------------------------------------------------------- #
    st.subheader("🚀 Train Model")

    if st.button("Start Training", type="primary", key="train_btn"):
        progress_bar = st.progress(0, text="Preparing dataset...")
        epoch_placeholder = st.empty()
        metric_cols = st.columns(4)
        m_epoch = metric_cols[0].empty()
        m_loss = metric_cols[1].empty()
        m_acc = metric_cols[2].empty()
        m_val_acc = metric_cols[3].empty()
        log_box = st.empty()

        def on_epoch_end_hook(progress: TrainingProgress):
            # Runs on the MAIN thread (fit() is called synchronously below),
            # so touching Streamlit widgets here is safe and renders live.
            frac = min(1.0, progress.current_epoch / max(progress.total_epochs, 1))
            progress_bar.progress(frac, text=f"Epoch {progress.current_epoch}/{progress.total_epochs}")
            m_epoch.metric("Epoch", f"{progress.current_epoch}/{progress.total_epochs}")
            m_loss.metric("Loss", f"{progress.current_loss:.4f}")
            m_acc.metric("Accuracy", f"{progress.current_acc:.4f}")
            m_val_acc.metric("Val Accuracy", f"{progress.current_val_acc:.4f}")
            log_box.text(
                f"Epoch {progress.current_epoch}/{progress.total_epochs} | "
                f"loss={progress.current_loss:.4f} acc={progress.current_acc:.4f} | "
                f"val_loss={progress.current_val_loss:.4f} val_acc={progress.current_val_acc:.4f}"
            )

        try:
            with st.spinner("Training in progress — watch live metrics above..."):
                t0 = time.time()
                model, history, progress, run_dir, bundle = run_training(
                    cfg,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    on_epoch_end_hook=on_epoch_end_hook,
                )
                elapsed = time.time() - t0

            progress_bar.progress(1.0, text="Training complete ✅")
            st.success(f"Training completed in {elapsed:.1f}s "
                       f"({progress.current_epoch} epochs run).")

            # ------------------------------------------------------------- #
            # 5. Final metrics + graphs
            # ------------------------------------------------------------- #
            st.subheader("📈 Final Metrics")
            fc1, fc2, fc3, fc4 = st.columns(4)
            fc1.metric("Final Train Acc", f"{progress.current_acc:.3f}")
            fc2.metric("Final Val Acc", f"{progress.current_val_acc:.3f}")
            fc3.metric("Final Train Loss", f"{progress.current_loss:.3f}")
            fc4.metric("Final Val Loss", f"{progress.current_val_loss:.3f}")

            st.subheader("📉 Accuracy / Loss Curves")
            fig = plot_training_curves(history)
            st.pyplot(fig)

            st.subheader("🔢 Confusion Matrix (validation set)")
            import numpy as np
            X_val_input = bundle.X_val[..., np.newaxis].astype("float32")  # (N, n_mfcc, T) -> (N, n_mfcc, T, 1)
            val_preds_raw = model.predict(X_val_input, verbose=0)
            if val_preds_raw.shape[-1] == 1:
                y_pred = (val_preds_raw.reshape(-1) >= 0.5).astype(int)
            else:
                y_pred = np.argmax(val_preds_raw, axis=1)
            cm_fig = plot_confusion_matrix(bundle.y_val, y_pred, cfg.classes)
            st.pyplot(cm_fig)

            # ------------------------------------------------------------- #
            # 6. Download model
            # ------------------------------------------------------------- #
            st.subheader("💾 Download Trained Model")
            model_path = cfg.models_path / "echoambuguard_model.keras"
            with open(model_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Model (.keras)",
                    data=f.read(),
                    file_name="echoambuguard_model.keras",
                    mime="application/octet-stream",
                )
            meta_path = cfg.models_path / "metadata.json"
            with open(meta_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Metadata (.json)",
                    data=f.read(),
                    file_name="metadata.json",
                    mime="application/json",
                )

        except Exception as e:
            progress_bar.progress(0, text="Training failed ❌")
            st.error(f"Training failed: {e}")
            logger.exception("[training_ui] Training raised an exception")
            st.exception(e)
