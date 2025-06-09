import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid


def load_data():

    X = np.load('data\X.npy')
    y = np.load('data\y.npy')

    return X, y

def build_model(lambda_):

    model = Sequential(
        [
            tf.keras.Input(shape=(400,)),
            Dense(units=25, activation='relu', name='L1', kernel_regularizer= tf.keras.regularizers.l2(lambda_)),
            Dense(units=15, activation='relu', name='L2', kernel_regularizer= tf.keras.regularizers.l2(lambda_)),
            Dense(units=10, activation='linear', name='L3')

        ], name = "my_model"
    )

    return model

def display_predictions_grid(X, y, model, title="Label, yhat", sample_size=64):
    """
    Displays a grid of 20x20 grayscale images with their true and predicted labels.
    
    Parameters:
    - X: Input feature matrix of shape (m, 400)
    - y: True labels of shape (m, 1) or (m,)
    - model: Trained Keras model
    - title: Title of the figure
    - sample_size: Number of images to display (should be a square number like 64, 49, etc.)
    """
    m, n = X.shape
    side = int(np.sqrt(sample_size))
    assert side * side == sample_size, "sample_size must be a perfect square (e.g., 64, 49)"

    fig, axes = plt.subplots(side, side, figsize=(6, 6))
    fig.tight_layout(pad=0.3, rect=[0, 0.03, 1, 0.93])

    for i, ax in enumerate(axes.flat):
        # Select a random index
        random_index = np.random.randint(m)

        # Reshape the image
        X_img = X[random_index].reshape((20, 20)).T

        # Display the image
        ax.imshow(X_img, cmap='gray')

        # Predict and apply softmax
        prediction = model.predict(X[random_index].reshape(1, -1), verbose=0)
        prediction_p = tf.nn.softmax(prediction)
        yhat = np.argmax(prediction_p)

        # Get true label
        true_label = y[random_index] if y.ndim == 1 else y[random_index, 0]

        # Set the title and turn off axis
        ax.set_title(f"{true_label},{yhat}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.show()
