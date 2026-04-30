import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load dataset
# -----------------------------
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# -----------------------------
# 2. Padding
# -----------------------------
max_len = 200

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# -----------------------------
# 3. Function to build model
# -----------------------------
def build_model(rnn_type):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=max_len))

    if rnn_type == "RNN":
        model.add(SimpleRNN(128))
    elif rnn_type == "LSTM":
        model.add(LSTM(128))
    elif rnn_type == "GRU":
        model.add(GRU(128))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# -----------------------------
# 4. Train Vanilla RNN
# -----------------------------
rnn_model = build_model("RNN")
rnn_model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=1)
rnn_loss, rnn_acc = rnn_model.evaluate(X_test, y_test)

# -----------------------------
# 5. Train LSTM
# -----------------------------
lstm_model = build_model("LSTM")
lstm_model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=1)
lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test)

# -----------------------------
# 6. Train GRU
# -----------------------------
gru_model = build_model("GRU")
gru_model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=1)
gru_loss, gru_acc = gru_model.evaluate(X_test, y_test)

# -----------------------------
# 7. Results Comparison
# -----------------------------
print("\n📊 Model Performance Comparison:")
print("--------------------------------")
print("Vanilla RNN Accuracy:", rnn_acc)
print("LSTM Accuracy:", lstm_acc)
print("GRU Accuracy:", gru_acc)