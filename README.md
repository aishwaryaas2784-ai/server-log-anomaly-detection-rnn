# Server Log Anomaly Detection Using RNN

## Project Overview
This project uses a Simple Recurrent Neural Network (SimpleRNN) to detect anomalous patterns in server logs. It is trained using the HDFS log dataset and provides an interactive Streamlit web application.

## Objectives
- Detect potentially anomalous server log patterns.
- Classify log blocks as Normal or Anomaly.
- Analyze sample logs, block IDs, custom event IDs, and uploaded log files.
- Display anomaly probability and prediction results.

## Technologies Used
- Python
- TensorFlow / Keras
- SimpleRNN
- Pandas and NumPy
- Scikit-learn
- Streamlit

## Dataset
The project uses the HDFS log dataset, which contains normal and anomalous log blocks. Dataset CSV files are not included in this repository.

## Model
The model uses an embedding layer, a SimpleRNN layer, dropout, and dense output layers to classify log-event sequences.

## Project Structure
```text
Server_Log_RNN/
├── app.py
├── train_model.py
├── inspect_data.py
├── requirements.txt
├── .gitignore
├── data/
└── models/
    ├── server_log_rnn.keras
    ├── event_classes.npy
    └── preprocessing.json
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aishwaryaas2784-ai/server-log-anomaly-detection-rnn.git
   cd server-log-anomaly-detection-rnn
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   ```

   Windows:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place the required HDFS dataset CSV files in the `data/` folder.

5. Train the model if needed:
   ```bash
   python train_model.py
   ```

6. Launch the application:
   ```bash
   streamlit run app.py
   ```

## Features
- Interactive Streamlit interface
- RNN-based anomaly prediction
- Block-level analysis
- Custom log input and log-file upload
- Anomaly probability display

## Important Note
This is an academic prototype. Predictions are estimates and should not be treated as definitive evidence of a real server incident. Performance may vary, especially for anomalous logs.

## Author
Aishwarya A S

## GitHub
https://github.com/aishwaryaas2784-ai/server-log-anomaly-detection-rnn