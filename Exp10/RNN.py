import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load dataset
# -----------------------------
vocab_size = 10000  # top 10k words

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# -----------------------------
# 2. Pad sequences
# -----------------------------
max_len = 200

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# -----------------------------
# 3. Build RNN (LSTM) model
# -----------------------------
model = Sequential()

# Embedding layer
model.add(Embedding(vocab_size, 128, input_length=max_len))

# LSTM layer
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# -----------------------------
# 4. Compile model
# -----------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 5. Train model
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# -----------------------------
# 6. Evaluate model
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print("\nTest Accuracy:", acc)

# -----------------------------
# 7. Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))