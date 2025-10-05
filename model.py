from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_bidir_lstm(input_shape: Tuple[int, int], horizon: int = 10, lr: float = 1e-3) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)  # (lookback, n_feats)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.2)(x)

    # Dense head
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(horizon, name='next_10_closes')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
    )
    return model

def default_callbacks(patience: int = 5):
    return [
        callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(1,patience//2), min_lr=1e-6)
    ]