
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "server_log_rnn.keras"
META_PATH = BASE / "models" / "preprocessing.json"
LOG_PATH = BASE / "data" / "HDFS_100k.log_structured.csv"
LABEL_PATH = BASE / "data" / "HDFS.anomaly_label.csv"

st.set_page_config(
    page_title="Server Log Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_dataset():
    logs = pd.read_csv(LOG_PATH)
    labels = pd.read_csv(LABEL_PATH)

    label_map = dict(zip(
        labels["BlockId"].astype(str),
        labels["Label"].astype(str)
    ))

    block_events = {}
    block_pattern = r"blk_-?\d+"

    for content, event_id in zip(logs["Content"], logs["EventId"]):
        for block_id in dict.fromkeys(
            re.findall(block_pattern, str(content))
        ):
            if block_id in label_map:
                block_events.setdefault(block_id, []).append(str(event_id))

    return logs, label_map, block_events


def encode_events(events, metadata):
    vocabulary = metadata["event_to_id"]
    max_len = int(metadata["max_len"])

    sequence = [
        vocabulary[event]
        for event in events
        if event in vocabulary
    ]

    if not sequence:
        return None

    return pad_sequences(
        [sequence],
        maxlen=max_len,
        padding="post",
        truncating="pre",
        value=0
    )


def predict_events(events, model, metadata):
    encoded = encode_events(events, metadata)

    if encoded is None:
        return None, "No recognized event IDs were found in the trained vocabulary."

    probability = float(model.predict(encoded, verbose=0)[0][0])
    threshold = float(metadata["threshold"])
    prediction = "Anomaly" if probability >= threshold else "Normal"

    return probability, prediction


st.title("🛡️ Server Log Anomaly Detection")
st.caption("SimpleRNN-based anomaly detection using the HDFS log dataset.")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Model files are missing. Run python train_model.py first.")
    st.stop()

model = load_model()
metadata = load_metadata()

mode = st.sidebar.radio(
    "Choose input",
    [
        "Analyze sample logs",
        "Analyze a block ID",
        "Check custom log text",
        "Upload a log file"
    ]
)

if mode in ["Analyze sample logs", "Analyze a block ID"]:
    logs, label_map, block_events = load_dataset()

    valid_blocks = [
        block_id for block_id, events in block_events.items()
        if events and label_map.get(block_id) in ("Normal", "Anomaly")
    ]

    if not valid_blocks:
        st.error("No labeled blocks could be matched to the dataset.")
        st.stop()

    if mode == "Analyze sample logs":
        count = st.sidebar.slider(
            "Number of blocks to analyze",
            min_value=10,
            max_value=min(200, len(valid_blocks)),
            value=min(50, len(valid_blocks))
        )

        if st.button("Analyze logs", type="primary"):
            results = []

            for block_id in valid_blocks[:count]:
                probability, prediction = predict_events(
                    block_events[block_id], model, metadata
                )

                if probability is not None:
                    results.append({
                        "Block ID": block_id,
                        "Actual label": label_map[block_id],
                        "Anomaly probability (%)": round(probability * 100, 2),
                        "Predicted label": prediction
                    })

            if results:
                result_df = pd.DataFrame(results)
                st.metric("Analyzed blocks", len(result_df))
                st.metric(
                    "Predicted anomalies",
                    int((result_df["Predicted label"] == "Anomaly").sum())
                )
                st.dataframe(result_df, use_container_width=True)

    else:
        block_id = st.selectbox("Select a block ID", valid_blocks)

        if st.button("Check block", type="primary"):
            probability, prediction = predict_events(
                block_events[block_id], model, metadata
            )

            if probability is not None:
                st.metric("Anomaly probability", f"{probability:.2%}")
                if prediction == "Anomaly":
                    st.error("RNN prediction: Anomaly")
                else:
                    st.success("RNN prediction: Normal")

                st.write("Actual dataset label:", label_map[block_id])

                matching_rows = logs[
                    logs["Content"].astype(str).str.contains(
                        re.escape(block_id), regex=True
                    )
                ]

                st.dataframe(
                    matching_rows[
                        ["Date", "Time", "Level", "Component", "Content", "EventId"]
                    ].head(100),
                    use_container_width=True
                )

else:
    if mode == "Check custom log text":
        log_text = st.text_area(
            "Enter event IDs or log lines",
            placeholder="E5\nE10\nE26",
            height=180
        )
    else:
        uploaded = st.file_uploader(
            "Upload a .log or .txt file",
            type=["log", "txt"]
        )

        log_text = ""
        if uploaded is not None:
            try:
                log_text = uploaded.getvalue().decode("utf-8", errors="replace")
            except Exception as exc:
                st.error(f"Could not read file: {exc}")

    if st.button("Analyze input", type="primary"):
        if not log_text.strip():
            st.warning("Enter log text or upload a non-empty file.")
        else:
            # This version supports event IDs already known to the model.
            events = re.findall(
                r"\bE\d+\b",
                log_text,
                flags=re.IGNORECASE
            )
            events = [event.upper() for event in events]

            probability, prediction = predict_events(
                events, model, metadata
            )

            if probability is None:
                st.warning(
                    "No recognized trained event IDs found. "
                    "This model currently supports HDFS event IDs such as E5 or E10; "
                    "arbitrary log formats need a parser."
                )
            else:
                st.metric("Anomaly probability", f"{probability:.2%}")

                if prediction == "Anomaly":
                    st.error("RNN prediction: Anomaly")
                else:
                    st.success("RNN prediction: Normal")

st.divider()
st.caption(
    "Academic prototype. Predictions are model estimates, "
    "not proof of an incident. Unrecognized log formats require preprocessing."
)
