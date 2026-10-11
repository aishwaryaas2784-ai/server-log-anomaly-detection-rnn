
import os
import re
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dropout, Dense, Masking

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_FILE = "data/HDFS_100k.log_structured.csv"
LABEL_FILE = "data/HDFS.anomaly_label.csv"
MODEL_DIR = "models"

MAX_LEN = 40
MAX_BLOCKS = 20000
EPOCHS = 10
BATCH_SIZE = 64

os.makedirs(MODEL_DIR, exist_ok=True)


def extract_block_ids(text):
    """Extract all distinct HDFS block IDs from a log message."""
    return list(dict.fromkeys(
        re.findall(r"blk_-?\d+", str(text))
    ))


def main():
    print("Loading HDFS datasets...")

    logs = pd.read_csv(DATA_FILE)
    labels = pd.read_csv(LABEL_FILE)

    required_logs = {"Content", "EventId"}
    required_labels = {"BlockId", "Label"}

    if not required_logs.issubset(logs.columns):
        raise ValueError(f"Log CSV must contain: {required_logs}")

    if not required_labels.issubset(labels.columns):
        raise ValueError(f"Label CSV must contain: {required_labels}")

    labels["BlockId"] = labels["BlockId"].astype(str)
    label_map = dict(zip(labels["BlockId"], labels["Label"]))

    # Map each event to every block ID explicitly mentioned in its content.
    block_events = {}

    for content, event_id in zip(logs["Content"], logs["EventId"]):
        block_ids = extract_block_ids(content)

        for block_id in block_ids:
            if block_id in label_map:
                block_events.setdefault(block_id, []).append(str(event_id))

    # Retain only blocks with known labels and at least one event.
    block_ids = [
        block_id for block_id, events in block_events.items()
        if events and label_map.get(block_id) in ("Normal", "Anomaly")
    ]

    if not block_ids:
        raise ValueError(
            "No labeled blocks matched. Check the dataset and block ID format."
        )

    # Use a reproducible subset to keep training practical.
    random.shuffle(block_ids)
    block_ids = block_ids[:MAX_BLOCKS]

    print("Matched blocks:", len(block_ids))
    print("Preparing event sequences...")

    # Event IDs become integer tokens; 0 is reserved for padding.
    event_names = sorted({
        event
        for block_id in block_ids
        for event in block_events[block_id]
    })
    event_to_id = {event: i + 1 for i, event in enumerate(event_names)}

    sequences = []
    targets = []
    kept_block_ids = []

    for block_id in block_ids:
        sequence = [
            event_to_id[event]
            for event in block_events[block_id]
        ]

        if not sequence:
            continue

        # Keep the most recent events for long sequences.
        sequence = sequence[-MAX_LEN:]

        sequences.append(sequence)
        targets.append(1 if label_map[block_id] == "Anomaly" else 0)
        kept_block_ids.append(block_id)

    X = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="pre",
        value=0
    )
    y = np.asarray(targets, dtype=np.int32)

    print("Sequence array shape:", X.shape)
    print("Normal blocks:", int(np.sum(y == 0)))
    print("Anomalous blocks:", int(np.sum(y == 1)))

    if len(np.unique(y)) < 2:
        raise ValueError("Training requires both Normal and Anomaly examples.")

    # Stratification keeps both classes represented in the splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        random_state=SEED,
        stratify=y_train
    )

    # Compensate for the imbalance between normal and anomalous blocks.
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

    vocab_size = len(event_to_id) + 1

    # Genuine recurrent neural network for ordered log event sequences.
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
    print("\nTraining RNN...")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    print("\nEvaluating on held-out test data...")
    probabilities = model.predict(X_test, verbose=0).ravel()

    # Threshold is configurable; evaluate the trade-off rather than
    # assuming accuracy alone is sufficient.
    threshold = 0.5
    predictions = (probabilities >= threshold).astype(int)

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    model_path = os.path.join(MODEL_DIR, "server_log_rnn.keras")
    model.save(model_path)

    # Save the vocabulary and settings so inference can use the same
    # event-ID mapping and sequence length.
    metadata = {
        "event_to_id": event_to_id,
        "max_len": MAX_LEN,
        "threshold": threshold,
        "padding_value": 0,
        "model_type": "SimpleRNN",
        "classes": {"0": "Normal", "1": "Anomaly"}
    }

    with open(
        os.path.join(MODEL_DIR, "preprocessing.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print(f"\nModel saved: {model_path}")
    print("Preprocessing metadata saved: models/preprocessing.json")
    print("RNN training completed.")


if __name__ == "__main__":
    main()
