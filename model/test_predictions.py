import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from CNN import custom_loss

# Load the trained model
model = load_model("model/trained_model.h5", custom_objects={'custom_loss': custom_loss})

# Load test data and labels
test_data = np.load("data_simulation/processed_data/test_data.npy", allow_pickle=True)
test_labels = np.load("data_simulation/processed_data/test_labels.npy", allow_pickle=True)

# Reshape test labels to match the output layer
num_outputs = 5 * 4
test_labels_reshaped = test_labels.reshape((-1, num_outputs))

# Make predictions on the test data
predictions = model.predict(test_data)

# Flatten the arrays for mean_squared_error
test_labels_flat = test_labels_reshaped.flatten()
predictions_flat = predictions.flatten()

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(test_labels_flat, predictions_flat)
print(f"Mean Squared Error on Test Data: {mse}")

# print or analyze individual predictions and ground truth
for i in range(len(test_data)):
    print(f"Example {i + 1} - Predictions: {predictions[i]}, Ground Truth: {test_labels[i]}")
