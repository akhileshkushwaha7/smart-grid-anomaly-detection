import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def normalize(data):
    data = data.reshape(-1, 1)
    return scaler.fit_transform(data)

def create_sequences(data, seq_length=50):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)

import numpy as np

def inject_attacks(data):
    data = data.copy()

    # spike attack
    spike_idx = np.random.randint(0, len(data), 100)
    data[spike_idx] = data[spike_idx] * 3

    # drop attack
    drop_idx = np.random.randint(0, len(data), 100)
    data[drop_idx] = data[drop_idx] * 0.2

    # noise attack (fixed)
    noise = np.random.normal(0, 0.5, size=data.shape)
    data = data + noise

    return data