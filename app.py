
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "server_log_rnn.keras"
METADATA_PATH = BASE_DIR / "models" / "preprocessing.json"
DATA_DIR = BASE_DIR / "data"

LOG_FILE = DATA_DIR / "HDFS_100k.log_structured.csv"
LABEL_FILE = DATA_DIR / "HDFS.anomaly_label.csv"

st.set_page_config(
    page_title="Server Log Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL AND METADATA
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )
    return tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    metadata = load_metadata()
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()


EVENT_TO_ID = metadata.get("event_to_id", {})
MAX_LEN = int(metadata.get("max_len", 40))
THRESHOLD = float(metadata.get("threshold", 0.5))


# --------------------------------------------------
# DATASET HELPERS
# --------------------------------------------------
@st.cache_data
def load_datasets():
    logs = None
    labels = None

    if LOG_FILE.exists():
        logs = pd.read_csv(LOG_FILE)

    if LABEL_FILE.exists():
        labels = pd.read_csv(LABEL_FILE)

    return logs, labels


def extract_block_ids(text):
    return list(dict.fromkeys(
        re.findall(r"blk_-?\d+", str(text))
    ))


def get_block_events(logs, block_id):
    """Find the ordered event sequence for a block."""
    if logs is None or "Content" not in logs.columns:
        return []

    if "EventId" not in logs.columns:
        return []

    events = []

    for _, row in logs.iterrows():
        content = str(row["Content"])

        if block_id in extract_block_ids(content):
            event = str(row["EventId"])

            if event in EVENT_TO_ID:
                events.append(event)

    return events


def extract_event_ids(text):
    """Extract supported event IDs from text or log lines."""
    found = re.findall(r"\bE\d+\b", str(text).upper())

    return [
        event for event in found
        if event in EVENT_TO_ID
    ]


# --------------------------------------------------
# PREDICTION HELPERS
# --------------------------------------------------
def predict_events(events):
    """
    Return anomaly probability and predicted class.

    The model output is interpreted as the probability
    of class 1 (Anomaly), as configured during training.
    """
    if not events:
        raise ValueError(
            "No supported event IDs were found."
        )

    sequence = [
        EVENT_TO_ID[event]
        for event in events
        if event in EVENT_TO_ID
    ][-MAX_LEN:]

    if not sequence:
        raise ValueError(
            "The input contains no event IDs known to the model."
        )

    padded = pad_sequences(
        [sequence],
        maxlen=MAX_LEN,
        padding="post",
        truncating="pre",
        value=0
    )

    probability = float(
        model.predict(padded, verbose=0).reshape(-1)[0]
    )

    prediction = (
        "Anomaly" if probability >= THRESHOLD else "Normal"
    )

    return probability, prediction


def show_prediction(probability, prediction):
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Anomaly probability",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric("RNN prediction", prediction)

    if prediction == "Anomaly":
        st.error(
            "⚠️ The model predicts an anomalous log pattern."
        )
    else:
        st.success(
            "✅ The model predicts a normal log pattern."
        )

    st.caption(
        "This is a model estimate, not proof of a security incident."
    )


def show_event_table(events):
    st.subheader("Recognized event sequence")
    st.dataframe(
        pd.DataFrame({
            "Position": range(1, len(events) + 1),
            "Event ID": events
        }),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🛡️ Log Analyzer")

input_mode = st.sidebar.radio(
    "Choose input",
    [
        "Analyze sample logs",
        "Analyze a block ID",
        "Check custom log text",
        "Upload a log file"
    ]
)

st.sidebar.divider()

st.sidebar.write("**Model information**")
st.sidebar.write("Architecture: SimpleRNN")
st.sidebar.write(f"Sequence length: {MAX_LEN}")
st.sidebar.write(f"Decision threshold: {THRESHOLD:.2f}")
st.sidebar.write(f"Known event types: {len(EVENT_TO_ID)}")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------
st.title("🛡️ Server Log Anomaly Detection")
st.caption(
    "SimpleRNN-based anomaly detection using the HDFS log dataset."
)


# --------------------------------------------------
# OPTION 1: SAMPLE LOGS
# --------------------------------------------------
if input_mode == "Analyze sample logs":

    st.header("Analyze sample logs")

    try:
        logs, labels = load_datasets()
    except Exception as exc:
        st.error(f"Unable to read dataset files: {exc}")
        st.stop()

    if logs is None or labels is None:
        st.warning(
            "Dataset files were not found. Place "
            "HDFS_100k.log_structured.csv and "
            "HDFS.anomaly_label.csv inside the data folder."
        )
    else:
        required_log_columns = {"Content", "EventId"}
        required_label_columns = {"BlockId", "Label"}

        if not required_log_columns.issubset(logs.columns):
            st.error("The structured log CSV is missing required columns.")
        elif not required_label_columns.issubset(labels.columns):
            st.error("The anomaly label CSV is missing required columns.")
        else:
            labels["BlockId"] = labels["BlockId"].astype(str)
            label_map = dict(zip(labels["BlockId"], labels["Label"]))

            block_ids = []

            for content in logs["Content"].astype(str):
                for block_id in extract_block_ids(content):
                    if block_id in label_map:
                        block_ids.append(block_id)

            block_ids = list(dict.fromkeys(block_ids))

            if not block_ids:
                st.warning(
                    "No labeled block IDs were found in the log content."
                )
            else:
                max_blocks = min(500, len(block_ids))

                count = st.sidebar.slider(
                    "Number of blocks to analyze",
                    min_value=10 if max_blocks >= 10 else 1,
                    max_value=max_blocks,
                    value=min(50, max_blocks)
                )

                if st.button("Analyze logs", type="primary"):
                    results = []
                    progress = st.progress(0)
                    status = st.empty()

                    selected_blocks = block_ids[:count]

                    for index, block_id in enumerate(selected_blocks):
                        events = get_block_events(logs, block_id)

                        if events:
                            try:
                                probability, prediction = predict_events(events)

                                results.append({
                                    "Block ID": block_id,
                                    "Actual label": label_map.get(
                                        block_id, "Unknown"
                                    ),
                                    "Anomaly probability (%)": round(
                                        probability * 100, 2
                                    ),
                                    "Predicted label": prediction
                                })
                            except ValueError:
                                pass

                        progress.progress(
                            (index + 1) / len(selected_blocks)
                        )
                        status.write(
                            f"Analyzing block {index + 1} "
                            f"of {len(selected_blocks)}"
                        )

                    progress.empty()
                    status.empty()

                    if results:
                        result_df = pd.DataFrame(results)

                        st.subheader("Analysis results")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Analyzed blocks", len(result_df))
                        c2.metric(
                            "Predicted anomalies",
                            int(
                                (result_df["Predicted label"] == "Anomaly")
                                .sum()
                            )
                        )
                        c3.metric(
                            "Actual anomalies",
                            int(
                                (result_df["Actual label"] == "Anomaly")
                                .sum()
                            )
                        )

                        st.dataframe(
                            result_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download results as CSV",
                            result_df.to_csv(index=False).encode("utf-8"),
                            file_name="server_log_analysis.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(
                            "No blocks could be analyzed using the "
                            "event vocabulary stored with this model."
                        )


# --------------------------------------------------
# OPTION 2: BLOCK ID
# --------------------------------------------------
elif input_mode == "Analyze a block ID":

    st.header("Analyze a specific block")

    try:
        logs, labels = load_datasets()
    except Exception as exc:
        st.error(f"Unable to read datasets: {exc}")
        st.stop()

    if logs is None or labels is None:
        st.warning("Required HDFS dataset files are missing.")
    else:
        labels["BlockId"] = labels["BlockId"].astype(str)

        all_blocks = labels["BlockId"].tolist()
        block_id = st.selectbox(
            "Select a block ID",
            options=all_blocks
        )

        if st.button("Check block", type="primary"):
            events = get_block_events(logs, block_id)

            if not events:
                st.warning(
                    "No supported events were found for this block."
                )
            else:
                try:
                    probability, prediction = predict_events(events)
                    show_prediction(probability, prediction)

                    actual = labels.loc[
                        labels["BlockId"] == block_id, "Label"
                    ]

                    if not actual.empty:
                        st.write(
                            f"**Actual dataset label:** {actual.iloc[0]}"
                        )

                    show_event_table(events)

                    if "Content" in logs.columns:
                        matching_rows = logs[
                            logs["Content"].astype(str).apply(
                                lambda text: block_id in extract_block_ids(text)
                            )
                        ]

                        st.subheader("Related log records")
                        st.dataframe(
                            matching_rows,
                            use_container_width=True,
                            hide_index=True
                        )

                except Exception as exc:
                    st.error(f"Block analysis failed: {exc}")


# --------------------------------------------------
# OPTION 3: CUSTOM TEXT
# --------------------------------------------------
elif input_mode == "Check custom log text":

    st.header("Analyze custom log text")

    st.info(
        "Enter HDFS event IDs learned during training, such as E5 or E10, "
        "or paste structured HDFS log lines containing those event IDs."
    )

    custom_text = st.text_area(
        "Enter event IDs or log lines",
        value="E5",
        height=160,
        placeholder="Example: E5 E10 E22"
    )

    if st.button("Analyze input", type="primary"):
        events = extract_event_ids(custom_text)

        if not events:
            st.warning(
                "No recognized trained event IDs found. "
                "This model requires HDFS event IDs from its training "
                "vocabulary. Arbitrary log formats need a parser."
            )
        else:
            try:
                probability, prediction = predict_events(events)
                show_prediction(probability, prediction)
                show_event_table(events)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


# --------------------------------------------------
# OPTION 4: UPLOAD LOG FILE
# --------------------------------------------------
elif input_mode == "Upload a log file":

    st.header("Upload a log file")

    st.info(
        "Upload a .log or .txt file containing supported HDFS event IDs "
        "or event IDs embedded in structured log lines."
    )

    uploaded_file = st.file_uploader(
        "Choose a log file",
        type=["log", "txt"]
    )

    if uploaded_file is not None:

        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size:,} bytes")

        if st.button("Analyze uploaded file", type="primary"):

            try:
                log_text = uploaded_file.getvalue().decode(
                    "utf-8", errors="replace"
                )

                events = extract_event_ids(log_text)

                if not events:
                    st.warning(
                        "No recognized trained HDFS event IDs were found. "
                        "This model supports event IDs present in its "
                        "training vocabulary. Other log formats require "
                        "a parser before prediction."
                    )
                else:
                    probability, prediction = predict_events(events)

                    show_prediction(probability, prediction)

                    st.write(f"**Recognized events:** {len(events)}")
                    st.write(
                        f"**Events used for prediction:** "
                        f"{min(len(events), MAX_LEN)}"
                    )

                    show_event_table(events)

                    event_counts = (
                        pd.Series(events, name="Event ID")
                        .value_counts()
                        .rename_axis("Event ID")
                        .reset_index(name="Count")
                    )

                    st.subheader("Event frequency")
                    st.bar_chart(
                        event_counts.set_index("Event ID")
                    )

            except Exception as exc:
                st.error(f"File analysis failed: {exc}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Academic prototype using a SimpleRNN trained on HDFS log sequences. "
    "Predictions are estimates and should be validated before operational use."
)
