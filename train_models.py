import pandas as pd
import numpy as np
import random
import tensorflow as tf
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# Set random seeds
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


# Load cleaned dataset
df = pd.read_csv("dataset/emails_cleaned.csv")

X = df["text"]
y = df["label"]


#split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Convert text into sequences
tokenizer = Tokenizer(
    num_words=20000,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


# Pad sequences to a fixed length
MAX_LENGTH = 300

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


# Build LSTM model
lstm_model = Sequential([
    tf.keras.Input(shape=(MAX_LENGTH,)),

    Embedding(
        input_dim=20000,
        output_dim=128
    ),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])


#compile LSTM model
lstm_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

#stop training when validation performance stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=1,
    restore_best_weights=True
)

# Train LSTM
lstm_model.fit(
    X_train_pad,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)


# LSTM predictions
lstm_predictions_prob = lstm_model.predict(
    X_test_pad,
    verbose=0
)

lstm_predictions = (
    lstm_predictions_prob > 0.5
).astype(int).flatten()


# LSTM evaluation
lstm_accuracy = accuracy_score(y_test, lstm_predictions)
lstm_precision = precision_score(y_test, lstm_predictions)
lstm_recall = recall_score(y_test, lstm_predictions)
lstm_f1 = f1_score(y_test, lstm_predictions)
lstm_roc_auc = roc_auc_score(
    y_test,
    lstm_predictions_prob
)

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


nb_model = MultinomialNB()

nb_model.fit(
    X_train_tfidf,
    y_train
)

nb_predictions = nb_model.predict(X_test_tfidf)

nb_probabilities = nb_model.predict_proba(
    X_test_tfidf
)[:, 1]


# Naive Bayes evaluation
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions)
nb_recall = recall_score(y_test, nb_predictions)
nb_f1 = f1_score(y_test, nb_predictions)
nb_roc_auc = roc_auc_score(
    y_test,
    nb_probabilities
)


lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(
    X_train_tfidf,
    y_train
)

lr_predictions = lr_model.predict(X_test_tfidf)

lr_probabilities = lr_model.predict_proba(
    X_test_tfidf
)[:, 1]


# Logistic Regression evaluation
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predictions)
lr_recall = recall_score(y_test, lr_predictions)
lr_f1 = f1_score(y_test, lr_predictions)
lr_roc_auc = roc_auc_score(
    y_test,
    lr_probabilities
)

svd = TruncatedSVD(
    n_components=300,
    random_state=42
)

X_train_svd = svd.fit_transform(X_train_tfidf)
X_test_svd = svd.transform(X_test_tfidf)


# Logistic Regression using SVD features
lr_svd_model = LogisticRegression(
    max_iter=1000
)

lr_svd_model.fit(
    X_train_svd,
    y_train
)

lr_svd_predictions = lr_svd_model.predict(
    X_test_svd
)

lr_svd_probabilities = lr_svd_model.predict_proba(
    X_test_svd
)[:, 1]


# Logistic Regression + SVD evaluation
lr_svd_accuracy = accuracy_score(
    y_test,
    lr_svd_predictions
)

lr_svd_precision = precision_score(
    y_test,
    lr_svd_predictions
)

lr_svd_recall = recall_score(
    y_test,
    lr_svd_predictions
)

lr_svd_f1 = f1_score(
    y_test,
    lr_svd_predictions
)

lr_svd_roc_auc = roc_auc_score(
    y_test,
    lr_svd_probabilities
)

comparison = pd.DataFrame({
    "Model": [
        "Naive Bayes",
        "Logistic Regression",
        "Logistic Regression + SVD",
        "LSTM"
    ],

    "Accuracy": [
        nb_accuracy,
        lr_accuracy,
        lr_svd_accuracy,
        lstm_accuracy
    ],

    "Precision": [
        nb_precision,
        lr_precision,
        lr_svd_precision,
        lstm_precision
    ],

    "Recall": [
        nb_recall,
        lr_recall,
        lr_svd_recall,
        lstm_recall
    ],

    "F1 Score": [
        nb_f1,
        lr_f1,
        lr_svd_f1,
        lstm_f1
    ],

    "ROC-AUC": [
        nb_roc_auc,
        lr_roc_auc,
        lr_svd_roc_auc,
        lstm_roc_auc
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(comparison)

# Save comparison results
comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

#saving LSTM model
lstm_model.save(
    "models/lstm_spam_detector.keras"
)


# Save tokenizer
with open("models/tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)


# Save traditional ML models
with open("models/naive_bayes.pkl", "wb") as file:
    pickle.dump(nb_model, file)

with open("models/logistic_regression.pkl", "wb") as file:
    pickle.dump(lr_model, file)

with open("models/logistic_regression_svd.pkl", "wb") as file:
    pickle.dump(lr_svd_model, file)


# Save preprocessing tools
with open("models/tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

with open("models/svd_transformer.pkl", "wb") as file:
    pickle.dump(svd, file)


print("\nAll models and preprocessing tools saved successfully!")