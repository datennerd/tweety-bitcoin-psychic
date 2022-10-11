"""Trainings pipeline: Hyperparameter optimization for new LSTM network."""

import config
from data import get_bitcoin_data
from model import build_model, create_window_dataset, select_keras_tuner
from utils import split_data, wrangle_data

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

print("\n[INFO] hyperparameter search...")
tuner = select_keras_tuner(build_model, config.KERAS_TUNER, "val_mae")
history = tuner.search(
    train_set,
    validation_data=val_set,
    epochs=config.EPOCHS_FOR_TUNER,
    verbose=2,
)

print("[INFO] get optimal hyperparameter...")
bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(bestHP)
model.summary()

print("[INFO] save suggested LSTM model...")
model.save(f"{config.OUTPUT_PATH}/lstm.h5")
