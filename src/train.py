from src.model import build_model

def train_model(X_train, seq_length):
    model = build_model(seq_length)

    history = model.fit(
        X_train, X_train,
        epochs=15,
        batch_size=32,
        validation_split=0.1
    )

    return model, history