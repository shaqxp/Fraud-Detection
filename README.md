# 🔐 Credit Card Fraud Detection

A machine learning application that detects fraudulent credit card transactions using a trained Logistic Regression model, served through an interactive Streamlit web app.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

---

## Overview

This project trains and deploys a binary classifier to identify whether a credit card transaction is **fraudulent** or **safe**. The model uses a probability threshold of **0.40** — transactions with a predicted fraud probability at or above this value are flagged as fraudulent.

---

## Dataset

The project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`creditcard.csv`).

- **284,807** transactions, of which **492 (0.17%)** are fraudulent.
- Features `V1` – `V28` are the result of a **PCA transformation** applied for anonymization. Their original business meaning is not disclosed.
- `Time`: Seconds elapsed since the first transaction in the dataset.
- `Amount`: Transaction amount.
- `Class`: Target label (`0` = Safe, `1` = Fraud).

> **Note:** The dataset file (`creditcard.csv`) is not included in this repository. Download it from Kaggle and place it in the project root before running the notebook.

---

## Project Structure

```
Fraud-Detection/
├── LOG_RF.ipynb        # Jupyter notebook for EDA, model training, and evaluation
├── app.py              # Streamlit web application
├── fraud_model.pkl     # Saved trained model (Logistic Regression)
├── scaler.pkl          # Saved StandardScaler for Time and Amount features
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Model Training

The notebook `LOG_RF.ipynb` covers the full ML pipeline:

1. **Data Loading & Exploration** – class distribution, shape inspection.
2. **Train/Test Split** – 80/20 split, stratified by class.
3. **Feature Scaling** – `StandardScaler` applied to `Time` and `Amount` only.
4. **Models Trained:**
   - Logistic Regression (`class_weight="balanced"`)
   - Random Forest (`n_estimators=100`, `class_weight="balanced"`)
5. **Evaluation** – Classification report, Confusion Matrix, ROC-AUC score.
6. **Export** – Final model (Logistic Regression) and scaler saved as `.pkl` files via `joblib`.

| Model               | Notes                          |
|---------------------|--------------------------------|
| Logistic Regression | Final model used in production |
| Random Forest       | Trained for comparison         |

**Decision threshold:** `0.40` (configurable in `app.py`)

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/shaqxp/Fraud-Detection.git
cd Fraud-Detection

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

### Using the App

1. Enter transaction values in the form (`Time`, `Amount`, and `V1`–`V28`).
2. Default values are pre-filled with a sample transaction.
3. Click **Check Transaction** to get the prediction.
4. The app displays:
   - ✅ **Safe Transaction** or 🚨 **Fraudulent Transaction Detected**
   - **Fraud Probability** (0.0 – 1.0)
   - **Risk Level**: Low (< 0.30), Medium (0.30 – 0.69), High (≥ 0.70)

### Retrain the Model

1. Place `creditcard.csv` in the project root.
2. Open and run `LOG_RF.ipynb` in Jupyter.
3. The notebook will regenerate `fraud_model.pkl` and `scaler.pkl`.

---

## Features

- Interactive Streamlit UI with a 3-column input form.
- Probability-based fraud detection with a configurable threshold.
- Three-tier risk level classification (Low / Medium / High).
- Feature description tooltips for all 30 input fields.
- Pre-filled sample transaction for quick testing.
