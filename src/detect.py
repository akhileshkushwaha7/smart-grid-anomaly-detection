import numpy as np

def detect_anomalies(model, X_test):
    X_pred = model.predict(X_test)

    mse = np.mean(np.power(X_test - X_pred, 2), axis=(1,2))

    return None, mse, None