"""Trainings pipeline: Retrain old LSTM network & Compare with new LSTM network."""

import os

import config
from clr_callback import CyclicLR
from data import get_bitcoin_data
from model import TimingCallback, create_window_dataset, early_stopping, reset_model
from tensorflow import keras
from utils import get_accuracy, split_data, wrangle_data

print("\n[INFO] load data...")
df = get_bitcoin_data()
print(f"...df: {df.shape}")

print("\n[INFO] wrangle data...")
df, scaler = wrangle_data(df, return_scaler=True, one_hot_encoding=False)

print("[INFO] split data...")
train_df, val_df, test_df = split_data(df)
print(f"...window size: {config.WINDOW_SIZE}")
print(
    f"...train_df: {train_df.shape}"
    f"\n...val_df: {val_df.shape}"
    f"\n...test_df: {test_df.shape}"
)

print("\n[INFO] create window datasets...")
train_set = create_window_dataset(train_df)
val_set = create_window_dataset(val_df)

if os.path.isdir("assets"):
    print("\n[INFO] load and retrain old lstm network...")
    model_old = keras.models.load_model("assets/lstm.h5")

    # Setup Cyclical Learning Rate (CLR)
    step_size = (8 - 2) * int(train_df.shape[0] / config.BATCH_SIZE)
    reco_lr = float(model_old.optimizer.lr)
    base_lr = reco_lr / config.MAX_LR_FACTOR
    max_lr = reco_lr * config.MAX_LR_FACTOR
    clr = CyclicLR(
        mode="triangular",
        base_lr=base_lr,
        max_lr=max_lr,
        step_size=step_size,
    )

    model_old = reset_model(model_old)
    model_old.summary()

    print("\n[INFO] start training...")
    timer = TimingCallback()
    history = model_old.fit(
        train_set,
        validation_data=val_set,
        epochs=config.EPOCHS_FOR_TRAINING,
        callbacks=[
            timer,
            early_stopping("val_mae"),
            clr,
        ],
        verbose=2,
    )

    print("\n[INFO] training statistics...")
    timer.statistics()

    print("\n[INFO] load and compare with new lstm networks...")
    model_new = keras.models.load_model(f"{config.OUTPUT_PATH}/lstm.h5")

    print("...calculation of the average mae for the test data:")
    mae_model_new = get_accuracy(model_new, test_df, scaler)
    mae_model_old = get_accuracy(model_old, test_df, scaler)
    print(f"...new model: {mae_model_new}")
    print(f"...old model: {mae_model_old}")

    if mae_model_new < mae_model_old:
        print("\n[INFO] Choose new lstm network for final training...")
    else:
        print("\n[INFO] Choose old lstm network for final training...")
        model_old.save(f"{config.OUTPUT_PATH}/lstm.h5")

else:
    print("\n[INFO] No old lstm network found...")
    model_new = keras.models.load_model(f"{config.OUTPUT_PATH}/lstm.h5")
    mae_model_new = get_accuracy(model_new, test_df, scaler)
    print(f"...choose new lstm network with an average mae of: {mae_model_new}")
