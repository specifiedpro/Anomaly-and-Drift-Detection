import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from typing import Dict, Any, Tuple, Optional


def build_autoencoder(
    input_dim: int,
    layer_dims: Optional[Tuple[int]] = (128, 64, 32),
    activation: str = 'leaky_relu',
    final_activation: str = 'sigmoid'
) -> keras.Model:
    """
    Build a generic Autoencoder model with symmetric encoder-decoder structure.

    :param input_dim: Number of input features.
    :param layer_dims: Tuple specifying the hidden layer sizes in the encoder (decoder is mirrored).
    :param activation: Activation function for hidden layers.
    :param final_activation: Activation for final output layer.
    :return: Compiled Keras model.
    """
    model = keras.Sequential()
    
    # Encoder part
    model.add(keras.layers.InputLayer(input_shape=(input_dim,)))
    for dim in layer_dims:
        model.add(keras.layers.Dense(dim, activation=activation))
    
    # Decoder part (mirror)
    for dim in reversed(layer_dims):
        model.add(keras.layers.Dense(dim, activation=activation))
    
    # Final layer
    model.add(keras.layers.Dense(input_dim, activation=final_activation))

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mean_squared_error',
        metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()]
    )
    
    return model


def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    model_path: str,
    layer_dims: Tuple[int] = (128, 64, 32),
    activation: str = 'leaky_relu',
    final_activation: str = 'sigmoid',
    n_epochs: int = 100,
    batch_size: int = 128,
    retrain: bool = False
) -> (keras.Model, Any):
    """
    Train an Autoencoder on the given data.

    :param X_train: Numpy array of training data.
    :param X_val: Numpy array of validation data.
    :param model_path: Where to save the best model checkpoint.
    :param layer_dims: Tuple specifying layer sizes.
    :param activation: Activation function for intermediate layers.
    :param final_activation: Activation for the final output layer.
    :param n_epochs: Number of epochs to train.
    :param batch_size: Batch size.
    :param retrain: If True, re-train from scratch; otherwise you might load existing weights if desired.
    :return: (model, history): The trained model and the training history object.
    """
    input_dim = X_train.shape[1]

    model = build_autoencoder(
        input_dim=input_dim,
        layer_dims=layer_dims,
        activation=activation,
        final_activation=final_activation
    )

    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    )

    history = model.fit(
        X_train,
        X_train,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        callbacks=[checkpoint],
        verbose=1
    )

    return model, history


def init_detector(
    data_dict: Dict[str, np.ndarray],
    model_path: str,
    default_weight: bool = True,
    hyper_params: Dict[str, Any] = None
) -> (keras.Model, Dict[str, np.ndarray]):
    """
    Initialize or load an Autoencoder drift detection model.

    :param data_dict: Dictionary with keys {'train', 'val', 'test'} -> arrays of data.
    :param model_path: Path to model checkpoint.
    :param default_weight: If True, load pre-trained model weights.
    :param hyper_params: Dictionary of training hyperparameters. Example keys:
                        {
                            'n_epochs': 30,
                            'batch_size': 128,
                            'layer_dims': (128, 64, 32),
                            'activation': 'leaky_relu',
                            'final_activation': 'sigmoid'
                        }
    :return: (detector, scaled_data) -> (Autoencoder model, dictionary with scaled train/val/test).
    """
    if hyper_params is None:
        hyper_params = {
            'n_epochs': 30,
            'batch_size': 128,
            'layer_dims': (128, 64, 32),
            'activation': 'leaky_relu',
            'final_activation': 'sigmoid'
        }

    if default_weight:
        # Load existing model from path
        detector = keras.models.load_model(model_path)
        print("[INFO] Loaded pre-trained AE model from:", model_path)
    else:
        # Train from scratch
        X_train = data_dict['train']
        X_val = data_dict['val']
        detector, _ = train_autoencoder(
            X_train=X_train,
            X_val=X_val,
            model_path=model_path,
            layer_dims=hyper_params['layer_dims'],
            activation=hyper_params['activation'],
            final_activation=hyper_params['final_activation'],
            n_epochs=hyper_params['n_epochs'],
            batch_size=hyper_params['batch_size'],
            retrain=False
        )
        print("[INFO] Trained a new AE model.")

    return detector, data_dict
