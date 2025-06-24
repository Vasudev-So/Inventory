import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st

# Constants
FEEDBACK_CSV = "feedback.csv"
MODEL_FILE = "sentiment_predictor.h5"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def analyze_sentiment(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = sentiment_model(**inputs).logits.numpy()[0]
    probs = softmax(logits)
    return float(probs[1])

def init_storage():
    if not os.path.exists(FEEDBACK_CSV):
        df = pd.DataFrame(columns=["timestamp", "user_id", "feedback", "sentiment"])
        df.to_csv(FEEDBACK_CSV, index=False, quoting=csv.QUOTE_MINIMAL)


def add_feedback(user_id: str, texts: list):
    rows = []
    for text in texts:
        sent = analyze_sentiment(text)
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "feedback": text,
            "sentiment": sent
        }
        rows.append(row)
    df = pd.read_csv(FEEDBACK_CSV)
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(FEEDBACK_CSV, index=False)

def prepare_sequences(max_history=10):
    import csv
    df = pd.read_csv(FEEDBACK_CSV, quoting=csv.QUOTE_MINIMAL, quotechar='"')

    sequences, targets = [], []

    for _, group in df.groupby("user_id"):
        sents = group.sort_values("timestamp")["sentiment"].tolist()
        for i in range(len(sents) - max_history):
            sequences.append(sents[i:i+max_history])
            targets.append(sents[i+max_history])

    if not sequences:
        return None, None

    return np.array(sequences), np.array(targets)

def build_and_train(X, y, max_history=10):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(max_history,)),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=100, batch_size=16, validation_split=0.1,
              callbacks=[es], verbose=0)

    model.save(MODEL_FILE)
    return model

def suggest_words(pred_score: float) -> list:
    positive_words = ["excellent", "amazing", "smooth", "intuitive", "satisfied"]
    neutral_words = ["okay", "fair", "adequate", "average", "sufficient"]
    negative_words = ["frustrated", "disappointed", "buggy", "slow", "unsatisfied"]

    if pred_score >= 0.7:
        return positive_words
    elif pred_score <= 0.3:
        return negative_words
    else:
        return neutral_words

def predict_next(user_id: str, model, max_history=10):
    df = pd.read_csv(FEEDBACK_CSV)
    user_sents = df[df.user_id == user_id].sort_values("timestamp")["sentiment"].tolist()
    if len(user_sents) < max_history:
        return None, user_sents, []
    seq = np.array(user_sents[-max_history:]).reshape(1, max_history)
    pred = float(model.predict(seq)[0][0])
    return pred, user_sents, suggest_words(pred)

# Streamlit UI
st.set_page_config(page_title="Sentiment Predictor", layout="centered")
st.title("\U0001F4AC Feedback Sentiment Tracker")
init_storage()

option = st.sidebar.selectbox("Select Action", ["Add Feedback", "Train Model", "Predict Next Sentiment"])

if option == "Add Feedback":
    st.header("Add Feedback")
    uid = st.text_input("User ID")
    feedbacks = st.text_area("Enter feedback (one per line)").splitlines()
    if st.button("Submit Feedback") and uid and feedbacks:
        add_feedback(uid, feedbacks)
        st.success("Feedback added and sentiment analyzed.")

elif option == "Train Model":
    st.header("Train Sentiment Prediction Model")
    X, y = prepare_sequences()
    if X is None:
        st.warning("Not enough data to train.")
    else:
        model = build_and_train(X, y)
        st.success("Model trained and saved.")

elif option == "Predict Next Sentiment":
    st.header("Predict Next Sentiment")
    uid = st.text_input("User ID")
    if st.button("Predict") and uid:
        if os.path.exists(MODEL_FILE):
            model = load_model(MODEL_FILE, compile=False)
            pred, history, suggestions = predict_next(uid, model)
            if pred is None:
                st.warning("Not enough history to predict.")
            else:
                st.markdown(f"**Predicted Sentiment Score:** {pred:.3f}")
                st.markdown(f"**Suggestions:** {', '.join(suggestions)}")
                st.line_chart(history[-10:])
        else:
            st.error("Model not trained yet. Please train the model first.")
