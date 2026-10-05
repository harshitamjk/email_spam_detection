import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load trad model
with open("models/naive_bayes.pkl", "rb") as file:
    nb_model = pickle.load(file)

with open("models/logistic_regression.pkl", "rb") as file:
    lr_model = pickle.load(file)

with open("models/logistic_regression_svd.pkl", "rb") as file:
    lr_svd_model = pickle.load(file)

with open("models/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("models/svd_transformer.pkl", "rb") as file:
    svd = pickle.load(file)

#load lstm
lstm_model = tf.keras.models.load_model(
    "models/lstm_spam_detector.keras"
)

with open("models/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

MAX_LENGTH = 300


print("All models loaded successfully!")


subject = input("\nEnter email subject: ")
body = input("Enter email body: ")

email_text = subject + " " + body


#trad ml
email_tfidf = vectorizer.transform([email_text])

nb_prediction = nb_model.predict(email_tfidf)[0]
nb_probability = nb_model.predict_proba(email_tfidf)[0][1]

lr_prediction = lr_model.predict(email_tfidf)[0]
lr_probability = lr_model.predict_proba(email_tfidf)[0][1]

email_svd = svd.transform(email_tfidf)

lr_svd_prediction = lr_svd_model.predict(email_svd)[0]
lr_svd_probability = lr_svd_model.predict_proba(email_svd)[0][1]

#lstm
email_sequence = tokenizer.texts_to_sequences([email_text])

email_padded = pad_sequences(
    email_sequence,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

lstm_probability = lstm_model.predict(
    email_padded,
    verbose=0
)[0][0]

lstm_prediction = 1 if lstm_probability >= 0.5 else 0

def label_name(prediction):
    if prediction == 1:
        return "SPAM"
    else:
        return "HAM"


print("\n======================================")
print("           MODEL COMPARISON")
print("======================================")

print(
    "\nNaive Bayes:"
    f"\nPrediction: {label_name(nb_prediction)}"
    f"\nSpam probability: {nb_probability * 100:.2f}%"
)

print(
    "\nLogistic Regression:"
    f"\nPrediction: {label_name(lr_prediction)}"
    f"\nSpam probability: {lr_probability * 100:.2f}%"
)

print(
    "\nLogistic Regression + SVD:"
    f"\nPrediction: {label_name(lr_svd_prediction)}"
    f"\nSpam probability: {lr_svd_probability * 100:.2f}%"
)

print(
    "\nLSTM:"
    f"\nPrediction: {label_name(lstm_prediction)}"
    f"\nSpam probability: {lstm_probability * 100:.2f}%"
)
