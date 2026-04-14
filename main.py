# =========================
# IMPORTS
# =========================

# We trained the model on normal sequences.
# During testing, we give attack data.
# The model does NOT “convert attack into normal intentionally”, but it tries to reconstruct it as if it were normal based on what it learned.
# Then we compare:

# attack input vs reconstructed output
# If the difference is large → anomaly is detected.
import numpy as np
import matplotlib.pyplot as plt

from src.load_data import load_nab_data
from src.preprocess import normalize, create_sequences, inject_attacks
from src.train import train_model
from src.detect import detect_anomalies


# =========================
# 1. LOAD DATA
# =========================
data, df = load_nab_data("data/nyc_taxi.csv")


# =========================
# 2. PREPROCESSING
# =========================
data_norm = normalize(data)


# =========================
# 3. CREATE NORMAL SEQUENCES (TRAIN DATA)
# =========================
seq_length = 50

X_normal = create_sequences(data_norm, seq_length)
X_normal = X_normal.reshape((X_normal.shape[0], X_normal.shape[1], 1))


# =========================
# 4. CREATE ATTACK DATA (TEST DATA)
# =========================
data_attack = inject_attacks(data_norm)

X_attack = create_sequences(data_attack, seq_length)
X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))


# =========================
# 5. TRAIN-TEST SPLIT
# =========================
split = int(0.8 * len(X_normal))
X_train = X_normal[:split]

# Test ONLY on attacked data
X_test = X_attack


# =========================
# 6. TRAIN MODEL
# =========================
model, history = train_model(X_train, seq_length)

# Save model
model.save("results/model.h5")


# =========================
# 7. DETECT ANOMALIES
# =========================
anomalies, mse, threshold = detect_anomalies(model, X_test)

# Improve threshold (override with percentile)
threshold = np.percentile(mse, 95)
anomalies = mse > threshold


# =========================
# 8. VISUALIZATIONS
# =========================

# 📈 Training Loss
plt.figure()
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.legend()
plt.title("Training Loss Curve")
plt.savefig("results/plots/training_loss.png")
plt.show()


# Reconstruction Error
plt.figure()
plt.plot(mse, label="Reconstruction Error")
plt.axhline(threshold, linestyle='--', label="Threshold")
plt.legend()
plt.title("Anomaly Detection")
plt.savefig("results/plots/anomaly_detection.png")
plt.show()


# Normal vs Attack Signal
plt.figure()
plt.plot(data_norm[:200], label="Normal")
plt.plot(data_attack[:200], label="Attacked")
plt.legend()
plt.title("Normal vs Attack Signal")
plt.savefig("results/plots/attack_simulation.png")
plt.show()


# Anomaly Points Highlight (IMPORTANT)
anomaly_points = np.where(anomalies)[0]

plt.figure()
plt.plot(mse, label="Reconstruction Error")
plt.scatter(anomaly_points, mse[anomaly_points], label="Anomalies")
plt.axhline(threshold, linestyle='--')
plt.legend()
plt.title("Detected Anomalies")
plt.savefig("results/plots/anomaly_points.png")
plt.show()


# =========================
# 9. METRICS
# =========================
print("\n===== RESULTS =====")
print("Threshold:", threshold)
print("Total anomalies detected:", np.sum(anomalies))
print("Percentage anomalies: {:.2f}%".format(np.mean(anomalies) * 100))


print("\n✅ Anomaly Detection Completed Successfully!")