import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load video dataset (frames)
# -----------------------------
def load_videos(folder, img_size=64, max_frames=10):
    X = []
    y = []
    labels = os.listdir(folder)

    for label_idx, label in enumerate(labels):
        path = os.path.join(folder, label)

        for video in os.listdir(path):
            video_path = os.path.join(path, video)

            cap = cv2.VideoCapture(video_path)
            frames = []

            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (img_size, img_size))
                frame = frame / 255.0
                frames.append(frame)

            cap.release()

            if len(frames) == max_frames:
                X.append(frames)
                y.append(label_idx)

    return np.array(X), np.array(y), labels

# -----------------------------
# 2. Load dataset (your path)
# -----------------------------
X, y, labels = load_videos("dataset")

y = to_categorical(y)

# -----------------------------
# 3. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Build CNN + LSTM model
# -----------------------------
model = Sequential()

# CNN applied on each frame
model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu'),
                          input_shape=(10, 64, 64, 3)))
model.add(TimeDistributed(MaxPooling2D((2,2))))
model.add(TimeDistributed(Flatten()))

# LSTM for sequence learning
model.add(LSTM(64))

# Output layer
model.add(Dense(len(labels), activation='softmax'))

# -----------------------------
# 5. Compile
# -----------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 6. Train
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# -----------------------------
# 7. Evaluate
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# -----------------------------
# 8. Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("Accuracy Score:", accuracy_score(y_true, y_pred_classes))