import numpy as np
import pandas as pd
import os 

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import warnings
warnings.filterwarnings("ignore")
from keras import backend as K

def train_model(X_train, X_val, model_path, n_epochs=100, batch_size=128, retrain=False):
    input_dim = X_train.shape[1]  # 輸入特徵的維度
    
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='leaky_relu',input_shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation='leaky_relu'),
    tf.keras.layers.Dense(32, activation='leaky_relu'),
    tf.keras.layers.Dense(64, activation='leaky_relu'),
    tf.keras.layers.Dense(128, activation='leaky_relu'),
    tf.keras.layers.Dense(input_dim, activation='sigmoid')
    ])
    
    # 編譯
    
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error', metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()])
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min')
    # 定義Early Stopping回調函數
    if retrain == False:
        model_history = model.fit(X_train, X_train, epochs=n_epochs, validation_data=(X_val, X_val), verbose=1,
                                       callbacks=[checkpoint])
    
#     model_history = model.fit(np.concatenate(X_train, X_val), np.concatenate(X_train, X_val), epochs=n_epochs, validation_data=(X_val, X_val), verbose=0)
    return model, model_history

def init_detector(data_dict:{}, default_weight=True, model_path='',hyper_params={'n_epochs':30, 'batch_size':128} ):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(data_dict['ref'])
    X_val = scaler.fit_transform(data_dict['val'])
    X_test = scaler.transform(data_dict['cur'])
    scaled_data = {'train':X_train, 'val':X_val, 'test':X_test}

    
    if default_weight == True: # If we already have trained the model for certain time period, then we don't have to train AE.
        detector = keras.models.load_model(model_path)
        return detector, scaled_data
    else: # Train the AE.
        
        detector, training_hist = train_model(X_train, X_val, model_path, 
                                              n_epochs=hyper_params['n_epochs'], 
                                              batch_size=hyper_params['batch_size'], 
                                              retrain=False)
    
    
        return detector, scaled_data
    