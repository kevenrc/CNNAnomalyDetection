import numpy as np
import tensorflow as tf
from model.CNN import create_model, custom_loss

# Example input shape (adjust based on your actual input)
input_shape = (100, 100, 1)

# Number of outputs per sphere (x, y, z, radius)
num_outputs = 4

model = create_model()

# Compile the model
model.compile(optimizer='adam', loss=custom_loss, metrics=['accuracy'])
#model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Loan Train Data
train_data = np.load("data_simulation/processed_data/train_data.npy", allow_pickle=True)
train_labels = np.load("data_simulation/processed_data/train_labels.npy", allow_pickle=True)

# Paramater tuning and cross validation eventually

#Fit the model
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.1)
# Save the trained model
model.save("model/trained_model.h5")
