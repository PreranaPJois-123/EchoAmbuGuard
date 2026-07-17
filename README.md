# EchoAmbuGuard — Ambulance Siren Detection

Rebuilt, production-hardened training pipeline for the "hangs at Epoch 1" bug.

## Root cause analysis

You verified TensorFlow works standalone with `np.random.rand(10,13,130,1)`.
That test *bypasses* your real pipeline (audio → MFCC → saved `.npy` →
loaded → CNN), so it cannot rule out the actual causes. The four most likely
culprits for "prints `Epoch 1/25`, never advances, even with the callback
removed" are:

1. **Inconsistent MFCC padding.** If even one saved feature file has a
   different time-frame count, stacking them into one array either fails
   silently (object array) or produces a model with a wildly different
   parameter count than expected — making the *first* CPU batch abnormally
   slow. → Fixed in `feature_extractor.pad_or_truncate()`, enforced again in
   `dataset_manager.validate_and_clean()`.
2. **Batch size larger than the validation split** (very likely with only
   5+5 samples) leaves Keras waiting on a validation batch that can never be
   filled. → Fixed by clamping `batch_size` to the actual sample counts in
   `dataset_manager.build_tf_datasets()`, and using `drop_remainder=False`
   with explicit `steps_per_epoch` / `validation_steps` so they are never 0.
3. **`verbose=1`** draws its progress bar with carriage returns (`\r`),
   which many terminals attached to `streamlit run` do not render — later
   epochs *were* printing, just invisibly. → Fixed by using `verbose=2`
   (one clean, flushed line per epoch) in `train_engine.run_training()`.
4. **A `Flatten → Dense` architecture** on a shape-drifted input can produce
   an enormous parameter count. → Fixed by using `GlobalAveragePooling2D`
   in `model_builder.py`, which is invariant to any residual shape drift and
   keeps the parameter count small and fast regardless.

Additionally, `model.fit()` is now always called **synchronously on
Streamlit's main thread** (never in a background `Thread`) — calling
`st.*` from a worker thread silently no-ops in Streamlit, which is a
separate, common cause of "nothing updates after epoch 1" reports.

## Project structure

```
EchoAmbuGuard/
├── app.py                 # Main Streamlit app (navigation)
├── training_ui.py          # Train Model page
├── config.json              # All tunables in one place
├── requirements.txt
├── src/
│   ├── utils.py             # config, logging, seeding, array sanitation
│   ├── feature_extractor.py # audio -> MFCC, corruption-safe
│   ├── dataset_manager.py   # load/validate/clean/split, tf.data pipelines
│   ├── model_builder.py     # CNN architecture (GlobalAveragePooling2D)
│   ├── train_engine.py      # robust, synchronous training orchestration
│   ├── plotting.py          # accuracy/loss curves, confusion matrix
│   └── prediction.py        # inference on new audio
├── dataset/ambulance/, dataset/non_ambulance/
├── features/                # saved X_features.npy / y_labels.npy
├── models/                  # saved .keras model, metadata.json, checkpoints
└── logs/                    # pipeline.log, tensorboard/, training_log.csv
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

1. Go to **Record / Upload Audio**, add clips to `ambulance` / `non_ambulance`.
2. Go to **Train Model** → click **Extract Features Now**, then **Start Training**.
3. Watch live epoch/loss/accuracy update, view final curves + confusion matrix,
   download the trained `.keras` model.
4. Go to **Predict** to classify a new clip.

## Notes on your 5+5 sample dataset

10 total samples is enough to prove the pipeline works end-to-end, but far
too small for a reliable model — expect the validation split to fall back
to 1 sample/class (`dataset_manager.safe_stratified_split` handles this
gracefully rather than erroring), and treat any accuracy number from this
size of dataset as not statistically meaningful. Add more samples per class
before trusting the model's predictions.
