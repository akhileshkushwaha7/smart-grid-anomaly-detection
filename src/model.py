from tensorflow.keras import layers, models

def build_model(seq_length):
    model = models.Sequential([
        layers.LSTM(64, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
        layers.LSTM(32, activation='relu', return_sequences=False),

        layers.RepeatVector(seq_length),

        layers.LSTM(32, activation='relu', return_sequences=True),
        layers.LSTM(64, activation='relu', return_sequences=True),

        layers.TimeDistributed(layers.Dense(1))
    ])

    model.compile(optimizer='adam', loss='mse')
    return model