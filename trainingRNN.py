import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load your data from a CSV file
df = pd.read_csv('coords1.csv')

# Extract features (X) and labels (y)
X = df.drop('class', axis=1).values  # Features
y = df['class'].values  # Labels

# Use LabelEncoder to convert string labels to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an RNN model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Reshape((X_train.shape[1], 1)),
    layers.LSTM(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y)), activation='softmax')  # Adjust output units based on your number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Save the trained model to a file in HDF5 format
model.save('rnn_model.h5')
