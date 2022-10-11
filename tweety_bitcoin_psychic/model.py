"""Hyperparameter tuner which will automatically select
the optimal values for the model to maximizes the accuracy.
"""

from timeit import default_timer as timer

import config
import tensorflow as tf
from keras_tuner import BayesianOptimization, Hyperband, RandomSearch
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import clone_model
from tensorflow.keras.optimizers import Adam


def early_stopping(metric_to_monitor, patience=config.PATIENCE):
    """Callback to prevent overfitting or spending the model.

    Args:
      metric_to_monitor: string
        Type of metric
      patience: integer
        Number of epochs with no improvement
        after which the training will be stopped.

    Returns:
      keras.callbacks.EarlyStopping
    """
    return EarlyStopping(
        monitor=metric_to_monitor,
        mode="min",
        patience=patience,
        restore_best_weights=True,
    )


def save_best_model(metric_to_monitor, model_name):
    """Save the Keras model or model weights at some frequency.

    Args:
      metric_to_monitor: string
        Type of metric
      model_name: string
        Name of saved model

    Returns:
      keras.callbacks.ModelCheckpoint
    """
    return ModelCheckpoint(
        f"{config.OUTPUT_PATH}/{model_name}",
        monitor=metric_to_monitor,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        verbose=1,
    )


class TimingCallback(Callback):
    """Callback which record the model training time."""

    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)

    def statistics(self):
        average_epoch_duration = round((sum(self.logs) / len(self.logs)) / 60, 2)
        print(f"...epochs trainined: {len(self.logs)}")
        print(f"...average epoch duration: {average_epoch_duration} minutes")
        print(f"...training duration: {round(sum(self.logs) / 60, 2)} minutes")


def select_keras_tuner(
    model,
    typ,
    metric_to_monitor,
    number_of_trials=config.TRIALS_FOR_TUNER,
    max_epochs=config.EPOCHS_FOR_TUNER,
):
    """Select type of KerasTuner for hyperparameter tuning.

    Args:
      model: keras.engine.functional.Functional
        Keras model
      typ: string
        Type of KerasTuner
      number_of_trials: integer
        Total number of trials
      number_of_trials: integer
        Maximum number of epochs to train one model

    Returns:
      keras_tuner.tuners
    """
    if typ == "Hyperband":
        tuner = Hyperband(
            model,
            objective=metric_to_monitor,
            max_epochs=max_epochs,
            factor=3,
            seed=42,
            directory=".",
            project_name="tmp/hyperband/",
        )

    elif typ == "RandomSearch":
        tuner = RandomSearch(
            model,
            objective=metric_to_monitor,
            max_trials=number_of_trials,
            seed=42,
            directory=".",
            project_name="tmp/random/",
        )

    elif typ == "BayesianOptimization":
        tuner = BayesianOptimization(
            model,
            objective=metric_to_monitor,
            max_trials=number_of_trials,
            seed=42,
            directory=".",
            project_name="tmp/bayesian/",
        )
    return tuner


def create_window_dataset(
    df,
    features=config.FEATURES,
    label=config.LABEL,
    window_size=config.WINDOW_SIZE,
    prediction_size=config.PREDICTION_SIZE,
    batch_size=config.BATCH_SIZE,
    shuffle_buffer=42,
):
    """Create a descriptive and efficient input pipeline for training.

    This function creates batches of features with a specific window size
    and corresponding labels with a specific prediction size.

    Args:
      df: pandas.core.frame.DataFrame
        Expect a DataFrame from data.get_bitcoin_data()
      features: list
        Features used for prediction
      label: string
        Feature to predict
      window_size: integer
        Feature window size for an prediction (Prefix)
      prediction_size: integer
        Size of one prediction
      batch_size: integer
        Length of one training example
      shuffle_buffer: integer
        Randomly samples elements from this buffer

    Returns:
      tensorflow.python.data.ops.dataset_ops.PrefetchDataset
    """
    # Create label: Single or multi-step
    label = tf.data.Dataset.from_tensor_slices(df[label])
    label = label.window(window_size + prediction_size, shift=1, drop_remainder=True)
    label = label.flat_map(lambda window: window.batch(window_size + prediction_size))
    label = label.map(
        lambda window: (window[:-prediction_size], window[-prediction_size:])
    )

    # Create feature: Univariate or multivariate
    feature_array = []
    for f in features:
        dataset = tf.data.Dataset.from_tensor_slices(df[f])
        dataset = dataset.window(window_size, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(window_size))
        feature_array.append(dataset)

    # Concat labels and featues and create shuffled batches
    dataset = tf.data.Dataset.zip((label, tuple(feature_array)))
    if len(features) > 1:
        dataset = dataset.map(lambda y, x: (tf.transpose(x), y[1]))
    else:
        dataset = dataset.map(lambda y, x: (tf.concat(x, axis=0), y[1]))
    dataset = dataset.shuffle(buffer_size=42)
    return dataset.batch(batch_size).prefetch(1)


def reset_model(model, metrics="mae"):
    """Creation of a model with new weights.

    Args:
      model: keras.engine.functional.Functional
        Trained TensorFlow model
      metrics: String
        Model metric

    Returns:
      keras.engine.functional.Functional
    """
    optimizer = model.optimizer.get_config()["name"]
    loss = model.loss
    model = clone_model(model)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def build_model(
    hp,
    features=config.FEATURES,
    window_size=config.WINDOW_SIZE,
    max_layers=config.MAX_LAYERS,
    prediction_size=config.PREDICTION_SIZE,
    loss=Huber(),
):
    """Create stacks of lstm and dropout layer.

    Args:
      features: list
        Features used for prediction
      window_size: integer
        Feature window size for an prediction (Prefix)
      max_layers: integer
        Max number of layer in network
      prediction_size: integer
        Size of one prediction
      loss: keras.losses.Huber

    Returns:
      keras.engine.functional.Functional
    """
    # Define inputs - i.e. [samples, timesteps, features]
    inputs = Input(shape=(window_size, len(features)))
    x = inputs

    num_layers = hp.Int("num_layers", 1, max_layers)
    for i in range(num_layers):

        # For Vanilla LSTM or last layer in Stacked LSTM
        if num_layers == 1 or i == num_layers - 1:
            x = LSTM(
                hp.Choice(f"lstm_{str(i)}", values=[16, 32, 64, 128, 256, 512]),
                activation=hp.Choice(
                    f"activation_{str(i)}",
                    values=["relu", "sigmoid"],
                ),
            )(x)

        # For first and hidden layer in stacked LSTM
        if i != num_layers - 1:
            x = LSTM(
                hp.Choice(f"lstm_with_rs_{str(i)}", values=[16, 32, 64, 128, 256, 512]),
                activation=hp.Choice(
                    f"activation_{str(i)}",
                    values=["relu", "sigmoid"],
                ),
                return_sequences=True,
            )(x)

        # Dropout layer
        x = Dropout(
            hp.Float(
                f"dropout_{str(i)}",
                min_value=0.0,
                max_value=0.3,
                step=0.1,
            )
        )(x)

    # Prediction layer
    x = Dense(prediction_size, activation="linear")(x)
    model = Model(inputs, x)

    # Initialize the learning rate, optimizer and loss
    optimizer = Adam(
        hp.Float(
            "learning_rate",
            min_value=1e-4,
            max_value=1e-2,
            sampling="log",
        )
    )

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
    return model
