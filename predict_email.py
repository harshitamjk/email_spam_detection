import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load trained LSTM model
model = tf.keras.models.load_model(
    "models/lstm_spam_detector.keras"
)

# Load tokenizer
with open("models/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

MAX_LENGTH = 300

print("Model and tokenizer loaded successfully!")


#user input
subject = input("\nEnter email subject: ")
body = input("Enter email body: ")

email_text = subject + " " + body


email_sequence = tokenizer.texts_to_sequences([email_text])
email_padded = pad_sequences(
    email_sequence,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


#spam probability
prediction_probability = model.predict(
    email_padded,
    verbose=0
)[0][0]

if prediction_probability >= 0.5:
    prediction = "SPAM"
else:
    prediction = "HAM"

print("\n========== PREDICTION ==========")
print("Prediction:", prediction)
print(
    "Spam probability:",
    round(prediction_probability * 100, 2),
    "%"
)