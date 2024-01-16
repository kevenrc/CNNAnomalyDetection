import tensorflow as tf
from tensorflow.keras import layers, models

class CustomOutputLayer(layers.Layer):
    def __init__(self, num_outputs, **kwargs):
        super(CustomOutputLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs

    def call(self, inputs):
        # Reshape the output to match the expected shape (batch_size, 5, 4)
        return tf.reshape(inputs, (-1, 5, self.num_outputs // 5, 4))



# Define the CNN-based model with an LSTM layer
def create_model(input_shape, num_outputs):
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())

    # Reshape the output for LSTM input
    model.add(layers.Reshape((-1, 32)))

    # Add an LSTM layer to capture temporal dependencies
    model.add(layers.LSTM(128, return_sequences=True))

    # Use a custom output layer to handle variable-sized outputs
    model.add(CustomOutputLayer(num_outputs))

    return model
