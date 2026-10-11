
import os
import re
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dropout, Dense

# ---------------- CONFIGURATION ----------------
SEED = 42
MAX_LEN = 40
MAX_BLOCKS = 20000
EPOCHS = 10
BATCH_SIZE = 64

DATA_FILE = "data/HDFS_100k.log_structured.csv"
LABEL_FILE = "data/HDFS.anomaly_label.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------- EXTRACT BLOCK IDS ----------------
def extract_block_ids(text):
    return list(dict.fromkeys(
        re.findall(r"blk_-?\d+", str(text))
    ))


# ---------------- MAIN TRAINING ----------------
def main():
    print("Loading HDFS datasets...")

    logs = pd.read_csv(DATA_FILE)
    labels = pd.read_csv(LABEL_FILE)

    if not {"Content", "EventId"}.issubset(logs.columns):
        raise ValueError("Log CSV must contain Content and EventId columns.")

    if not {"BlockId", "Label"}.issubset(labels.columns):
        raise ValueError("Label CSV must contain BlockId and Label columns.")

    labels["BlockId"] = labels["BlockId"].astype(str)
    label_map = dict(zip(labels["BlockId"], labels["Label"]))

    # Group event IDs by their HDFS block.
    block_events = {}

    for content, event_id in zip(logs["Content"], logs["EventId"]):
        for block_id in extract_block_ids(content):
            if block_id in label_map:
                block_events.setdefault(block_id, []).append(str(event_id))

    block_ids = [
        block_id for block_id, events in block_events.items()
        if events and label_map.get(block_id) in ("Normal", "Anomaly")
    ]

    if not block_ids:
        raise ValueError(
            "No labeled blocks matched. Check the dataset files."
        )

    random.shuffle(block_ids)
    block_ids = block_ids[:MAX_BLOCKS]

    print("Matched blocks:", len(block_ids))

    # Build event vocabulary.
    event_names = sorted({
        event
        for block_id in block_ids
        for event in block_events[block_id]
    })

    event_to_id = {
        event: index + 1
        for index, event in enumerate(event_names)
    }

    sequences = []
    targets = []
    kept_block_ids = []

    for block_id in block_ids:
        sequence = [
            event_to_id[event]
            for event in block_events[block_id]
        ][-MAX_LEN:]

        if sequence:
            sequences.append(sequence)
            targets.append(
                1 if label_map[block_id] == "Anomaly" else 0
            )
            kept_block_ids.append(block_id)

    X = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="pre",
        value=0
    )
    y = np.asarray(targets, dtype=np.int32)

    print("Sequence shape:", X.shape)
    print("Normal blocks:", int(np.sum(y == 0)))
    print("Anomalous blocks:", int(np.sum(y == 1)))

    if len(np.unique(y)) != 2:
        raise ValueError("Dataset must contain Normal and Anomaly examples.")

    # Split into training, validation and testing sets.
    indices = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.15,
        random_state=SEED,
        stratify=y[train_idx]
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Handle class imbalance.
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }

    # ---------------- SIMPLE RNN MODEL ----------------
    vocab_size = len(event_to_id) + 1

    model = Sequential([
        tf.keras.Input(shape=(MAX_LEN,), dtype="int32"),
        Embedding(
            input_dim=vocab_size,
            output_dim=32,
            mask_zero=True
        ),
        SimpleRNN(32),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    model.summary()
    print("\nTraining SimpleRNN...")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # ---------------- SELECT THRESHOLD ----------------
    print("\nSelecting threshold using validation data...")

    val_probabilities = model.predict(
        X_val, verbose=0
    ).ravel()

    best_threshold = 0.5
    best_f1 = -1.0

    for candidate in np.arange(0.10, 0.91, 0.05):
        val_predictions = (
            val_probabilities >= candidate
        ).astype(int)

        score = f1_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(candidate)

    threshold = best_threshold

    print(f"Selected threshold: {threshold:.2f}")
    print(f"Validation anomaly F1: {best_f1:.4f}")

    # ---------------- FINAL TEST EVALUATION ----------------
    print("\nEvaluating on held-out test data...")

    test_probabilities = model.predict(
        X_test, verbose=0
    ).ravel()

    test_predictions = (
        test_probabilities >= threshold
    ).astype(int)

    print("\nClassification report:")
    print(classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    ))

    # ---------------- SAVE MODEL ----------------
    model_path = os.path.join(
        MODEL_DIR, "server_log_rnn.keras"
    )
    model.save(model_path)

    metadata = {
        "event_to_id": event_to_id,
        "max_len": MAX_LEN,
        "threshold": threshold,
        "padding_value": 0,
        "model_type": "SimpleRNN",
        "classes": {
            "0": "Normal",
            "1": "Anomaly"
        }
    }

    metadata_path = os.path.join(
        MODEL_DIR, "preprocessing.json"
    )

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nModel saved:", model_path)
    print("Metadata saved:", metadata_path)
    print("RNN training completed successfully.")


if __name__ == "__main__":
    main()
