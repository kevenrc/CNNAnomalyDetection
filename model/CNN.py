import tensorflow as tf
from tensorflow.keras import layers, models

# custom loss function NOT WORKING
def custom_loss(y_true, y_pred):
    # Extract the actual coordinates, radii, and ground truth number of spheres
    actual_coords = y_true[:, :3]
    actual_radius = y_true[:, 3]
    mask_actual_coords = tf.not_equal(actual_coords, 999)
    mask_actual_radius = tf.not_equal(actual_radius, 999)
    num_actual_spheres = tf.reduce_sum(tf.cast(tf.reduce_all(mask_actual_coords, axis=1), dtype=tf.float32))

    # Extract the predicted coordinates, radii, and predicted number of spheres
    pred_coords = y_pred[:, :3]
    pred_radius = y_pred[:, 3]
    mask_pred = tf.greater(pred_radius, 700)
    num_pred_spheres = tf.reduce_sum(tf.cast(mask_pred, dtype=tf.float32))

    # Calculate count_loss
    count_loss = tf.square(num_actual_spheres - num_pred_spheres)

    # Apply masks for coordinate_loss
    masked_actual_coords = tf.boolean_mask(actual_coords, mask_actual_coords)
    masked_pred_coords = tf.boolean_mask(pred_coords, mask_actual_coords)
    
    # Calculate coordinate_loss
    coordinate_loss = tf.sqrt(tf.reduce_sum(tf.square(masked_actual_coords - masked_pred_coords)) / num_actual_spheres)

    # Apply masks for radius_loss
    masked_actual_radius = tf.boolean_mask(actual_radius, mask_actual_radius)
    masked_pred_radius = tf.boolean_mask(pred_radius, mask_actual_radius)
    
    # Calculate radius_loss
    radius_loss = tf.reduce_sum(tf.square(masked_actual_radius - masked_pred_radius)) / num_actual_spheres

    # Combine the losses with weights
    total_loss = 0.5 * coordinate_loss + 0.3 * radius_loss + 0.2 * count_loss

    return total_loss

# sort of off the shelf cnn - customize for better prediction
def create_model():
    # Define the CNN model
    model = models.Sequential()

    # Convolutional layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional layer 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten layer to feed into fully connected layers
    model.add(layers.Flatten())

    # Fully connected layer 1
    model.add(layers.Dense(256, activation='relu'))

    # Fully connected layer 2 (output layer)
    model.add(layers.Dense(5 * 4, activation='softmax'))

    # Reshape output to match the desired shape (5x4)
    model.add(layers.Reshape((5, 4)))

    return model