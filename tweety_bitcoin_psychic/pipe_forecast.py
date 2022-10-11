"""Trainings pipeline: Final model training."""

import config
from clr_callback import CyclicLR
from data import get_bitcoin_data
from model import (
    TimingCallback,
    create_window_dataset,
    early_stopping,
    reset_model,
    save_best_model,
)
from tensorflow import keras
from utils import plot_forecast, plot_training, split_data, wrangle_data

print("\n[INFO] load data...")
df = get_bitcoin_data()
print(f"...df: {df.shape}")

print("\n[INFO] wrangle data...")
df, scaler = wrangle_data(df, return_scaler=True, one_hot_encoding=False)

print("[INFO] split data...")
train_df, val_df = split_data(df, splits=config.TRAIN_VAL_SPLIT)
print(f"...window size: {config.WINDOW_SIZE}")
print(f"...train_df: {train_df.shape}\n...val_df: {val_df.shape}")

print("\n[INFO] create window datasets...")
train_set = create_window_dataset(train_df)
val_set = create_window_dataset(val_df)

print("[INFO] load LSTM model...")
model = keras.models.load_model(f"{config.OUTPUT_PATH}/lstm.h5")

# Setup Cyclical Learning Rate (CLR)
step_size = (8 - 2) * int(train_df.shape[0] / config.BATCH_SIZE)
reco_lr = float(model.optimizer.lr)
base_lr = reco_lr / config.MAX_LR_FACTOR
max_lr = reco_lr * config.MAX_LR_FACTOR
clr = CyclicLR(
    mode="triangular",
    base_lr=base_lr,
    max_lr=max_lr,
    step_size=step_size,
)

model = reset_model(model)
model.summary()

print("\n[INFO] start training...")
timer = TimingCallback()
history = model.fit(
    train_set,
    validation_data=val_set,
    epochs=config.EPOCHS_FOR_TRAINING,
    callbacks=[
        timer,
        early_stopping("val_mae"),
        save_best_model("val_mae", "lstm.h5"),
        clr,
    ],
    verbose=2,
)

print("\n[INFO] training statistics...")
timer.statistics()

print("\n[INFO] create training figures...")
plot_training(history, clr, reco_lr, save_fig=True)
plot_forecast(model, val_df, scaler, save_fig=True, save_csv=True)
