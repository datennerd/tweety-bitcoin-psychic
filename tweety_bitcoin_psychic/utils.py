"""Several helper functions to automate the model building."""

import os.path
from datetime import date, datetime, timedelta

import config
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import ticker
from pandas import DataFrame, Series, read_csv, to_datetime
from sklearn.preprocessing import MinMaxScaler


def wrangle_data(
    df,
    drop_actual_date=True,
    min_max_scaling=True,
    one_hot_encoding=True,
    return_scaler=False,
):
    """Convert a DataFrame into a better format for training a neural network.

    Args:
      df: pandas.core.frame.DataFrame
        Expect a DataFrame from data.get_bitcoin_data()
        any other Dataframe is also possible but then needs a list for
        min_max_scaling and/or one_hot_encoding.
      drop_actual_date: boolean
        True, drop the last row if is it the actual date
      min_max_scaling: boolean or list
        Name of feature columns
      one_hot_encoding: boolean or list
        Name of feature columns
      return_scaler: boolean
        Returns the MinMaxScaler instances

    Returns:
      pandas.core.frame.DataFrame,
      return_scaler: (optional)
        sklearn.preprocessing._data.MinMaxScaler
    """
    df = df.copy()

    if drop_actual_date:
        today = date.today().strftime("%Y-%m-%d %H:%M:%S")
        last_datetime = str(df.index[-1])
        if today == last_datetime:
            df = df[:-1]

    # Drop last row if it is not a full hour
    if df[-1:].index[0].strftime("%M%S") != "0000":
        df.drop(df.tail(1).index, inplace=True)

    # Standardize features
    if min_max_scaling is True or isinstance(min_max_scaling, list):
        if isinstance(min_max_scaling, list):
            COLUMNS_TO_SCALE = min_max_scaling

        elif min_max_scaling is True:
            COLUMNS_TO_SCALE = [
                "open",
                "high",
                "low",
                "close",
                "open_pchg",
                "high_pchg",
                "low_pchg",
                "close_pchg",
                "volume_pchg",
                "open_close_pdiff",
                "high_close_pdiff",
                "low_close_pdiff",
                "ma1",
                "ma7",
                "ma14",
            ]
            if "gtrend" in df.columns:
                COLUMNS_TO_SCALE.append("gtrend")

        scaler = MinMaxScaler(feature_range=(0, 1))
        df[COLUMNS_TO_SCALE] = scaler.fit_transform(df[COLUMNS_TO_SCALE])

    # One-Hot Encoding features
    if one_hot_encoding is True or isinstance(one_hot_encoding, list):
        if isinstance(one_hot_encoding, list):
            COLUMNS_TO_ONEHOT = one_hot_encoding

        elif one_hot_encoding is True:
            COLUMNS_TO_ONEHOT = ["pfc", "day_of_week"]

        for column in COLUMNS_TO_ONEHOT:
            num_classes = len(set(df[column]))
            df[column] = list(
                tf.keras.utils.to_categorical(
                    df[column], num_classes=num_classes, dtype="int32"
                )
            )

    if return_scaler:
        return df, scaler
    else:
        return df


def plot_distribution(df, features_to_drop=["pfc", "day_of_week"], save_fig=False):
    """Plot of the standardized features.

    Args:
      dataframe: pandas.core.frame.DataFrame
        Expect a DataFrame from data.wrangle_data()
      features_to_drop: list
        Features for exclude
      save_fig: boolean
        True, when plot should be saved

    Returns:
      plotly.graph_objs._figure.Figure
    """
    # Drop One-Hot Encoded features
    df = df.drop(features_to_drop, axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.violinplot(df)
    ax.set_title("Data distribution")
    ax.set_ylabel("Value range")
    ax.set_xlabel("Features")
    positions = list(range(1, len(df.columns) + 1))
    labels = df.columns
    ax.xaxis.set_major_locator(ticker.FixedLocator(positions))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    plt.xticks(rotation=45)

    if save_fig:
        fig.savefig(f"{config.OUTPUT_PATH}/distribution.png")
    plt.show()


def split_data(
    df, splits=config.TRAIN_VAL_TEST_SPLIT, prefix_window_size=config.WINDOW_SIZE
):
    """Split a DataFrame in train, validation and test set.

    Args:
      df: pandas.core.frame.DataFrame
      splits : float or list of floats
        The list can contain a single or two floats.
        Single float seperate the dataframe in a train and test set.
        Two floats seperate the dataframe in a train, validation and test set.
      window_size: integer
        Feature window size for an prediction (Prefix)

    Returns:
      pandas.core.frame.DataFrame
    """
    n = len(df)

    if isinstance(splits, float) or (isinstance(splits, list) and len(splits) == 1):
        if isinstance(splits, float):
            splits = [splits]

        train_df = df[0 : int(n * splits[0])]
        if prefix_window_size is not None:
            test_df = df[int(n * splits[0]) - prefix_window_size :]
        else:
            test_df = df[int(n * splits[0]) :]
        return train_df, test_df

    elif isinstance(splits, list) and len(splits) == 2:
        train_df = df[0 : int(n * splits[0])]
        if prefix_window_size is not None:
            val_df = df[int(n * splits[0]) - prefix_window_size : int(n * splits[1])]
            test_df = df[int(n * splits[-1]) - prefix_window_size :]
        else:
            val_df = df[int(n * splits[0]) : int(n * splits[1])]
            test_df = df[int(n * splits[-1]) :]
        return train_df, val_df, test_df


def get_subset_size(dataset, size=config.SUBSET_SIZE):
    """Calculates a number of iterations for a given percentage value.

    To minimize the time required to determine the learning rate,
    it may be useful not to iterate over the entire data set,
    but to use only a subset.

    Args:
      dataset: tensorflow.python.data.ops.dataset_ops.PrefetchDataset
        Dataset from create_window_dataset()
      size: float
        Percentage value

    Returns:
      Integer
    """
    dataset_to_numpy = list(dataset.as_numpy_iterator())
    num_iterations = len(dataset_to_numpy)
    subset_size = int(num_iterations * size)
    return subset_size


def plot_training(history, clr, reco_lr, save_fig=False):
    """Plot to evaluate the model training.

    Args:
      history: keras.callbacks.History
        History of model training
      clr: clr_callback.CyclicLR
        Learning rate history
      reco_lr: float
        KerasTuner detected learning rate
      save_fig: boolean
        True, when plot should be saved

    Returns:
      plotly.graph_objs._figure.Figure
    """
    plt.subplots(figsize=(14, 8))
    plt.subplot(3, 1, 1)
    plt.title("CLR, Loss and Accuracy during Training")
    plt.ylabel("CLR Policy")
    plt.plot(
        clr.history["iterations"], clr.history["lr"], label="Cyclical LR", color="green"
    )
    plt.axhline(y=reco_lr, label="Recommended LR", color="green", linestyle=":")
    plt.legend(loc="upper right")

    plt.subplot(3, 1, 2)
    plt.ylabel("Loss")
    plt.plot(history.epoch, history.history["loss"], label="train_loss", color="red")
    plt.plot(
        history.epoch,
        history.history["val_loss"],
        label="val_loss",
        color="red",
        linestyle="--",
    )
    plt.legend(loc="upper right")

    plt.subplot(3, 1, 3)
    plt.ylabel("Accuracy (MAE)")
    plt.xlabel("Training Iterations (CLR) and Epochs (Loss & Accuracy)")
    plt.plot(history.epoch, history.history["mae"], label="train_mae", color="blue")
    plt.plot(
        history.epoch,
        history.history["val_mae"],
        label="val_mae",
        color="blue",
        linestyle="--",
    )
    plt.legend(loc="upper right")

    if save_fig:
        plt.savefig(f"{config.OUTPUT_PATH}/training.png")
    plt.show()


def _get_prediction(
    model,
    df,
    features=config.FEATURES,
    label=config.LABEL,
    window_size=config.WINDOW_SIZE,
    prediction_size=config.PREDICTION_SIZE,
    include_label=True,
):
    """Get predictions from an input pipeline.

    Same function as model.create_window_dataset(), but with a predict
    on the end and without a shuffle.

    Args:
      model: keras.engine.functional.Functional
        Trained network
      df: pandas.core.frame.DataFrame
        DataFrame from wrangle_data()
      features: list
        Features used for prediction
      label: string
        Feature to predict
      window_size: integer
        Feature window size for an prediction (Prefix)
      prediction_size: integer
        Size of one prediction
      include_label: boolean
        True, for include the label

    Returns:
      numpy.ndarray
    """
    # Create label: Single or multi-step
    if include_label:
        label = tf.data.Dataset.from_tensor_slices(df[label])
        label = label.window(
            window_size + prediction_size, shift=1, drop_remainder=True
        )
        label = label.flat_map(
            lambda window: window.batch(window_size + prediction_size)
        )
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

    # Concat labels and featues and create batches
    if include_label:
        dataset = tf.data.Dataset.zip((label, tuple(feature_array)))
        if len(features) > 1:
            dataset = dataset.map(lambda y, x: (tf.transpose(x), y[1]))
        else:
            dataset = dataset.map(lambda y, x: (tf.concat(x, axis=0), y[1]))

    else:
        dataset = tf.data.Dataset.zip((tuple(feature_array), tuple(feature_array)))
        if len(features) > 1:
            dataset = dataset.map(lambda x, y: (tf.transpose(x)))
        else:
            dataset = dataset.map(lambda x, y: (tf.concat(x, axis=0)))
    dataset = dataset.batch(1).prefetch(1)
    return model.predict(dataset)


def _rescale_feature(prediction, scaler, feature=config.LABEL):
    """Rescaling a single feature.

    prediction: pandas.DataFrame or numpy.ndarray
      Expect Input from _get_prediction()
    scaler: sklearn.preprocessing._data.MinMaxScaler
      Expect scaler from wrangle_data()
    feature: string
      Single feature for rescaling

    Returns:
      pandas.DataFrame or pandas.Series
    """
    index = list(scaler.get_feature_names_out()).index(feature)
    return (
        prediction * (scaler.data_max_[index] - scaler.data_min_[index])
        + scaler.data_min_[index]
    )


def get_accuracy(
    model,
    df,
    scaler,
    label=config.LABEL,
    window_size=config.WINDOW_SIZE,
    prediction_size=config.PREDICTION_SIZE,
    return_plot=False,
    save_fig=False,
):
    """Calculates the MAE for a complete Dataframe.

    Args:
      model: keras.engine.functional.Functional
        Trained network
      df: pandas.core.frame.DataFrame
        DataFrame from wrangle_data()
      scaler: sklearn.preprocessing._data.MinMaxScaler
        Expect scaler from wrangle_data()
      label: string
        Feature to predict
      window_size: integer
        Feature window size for an prediction (Prefix)
      prediction_size: integer
        Size of one prediction
      return_plot: boolean
        True, returns a plot
      save_fig: boolean
        True, when the plot should be saved

    Returns:
      plotly.graph_objs._figure.Figure
    """
    if prediction_size > 1:
        predictions = _get_prediction(model, df)[:-1]
        original = df[label][window_size:-prediction_size]

        # Calculate the MAE for every timestep
        maes = []
        for i in range(len(predictions)):
            pred = _rescale_feature(predictions[i], scaler)
            start = config.WINDOW_SIZE + i
            end = config.WINDOW_SIZE + config.PREDICTION_SIZE + i
            orig = _rescale_feature(df[label][start:end], scaler)
            mae = tf.keras.metrics.mean_absolute_error(orig, pred).numpy()
            maes.append(mae)
            mae = round(sum(maes) / len(maes), 2)

    else:
        prediction = _get_prediction(model, df)[:, 0]
        prediction = _rescale_feature(prediction, scaler)
        original = _rescale_feature(df[label][window_size:], scaler)
        mae = round(
            float(tf.keras.metrics.mean_absolute_error(original, prediction).numpy()),
            2,
        )

    if not return_plot:
        return mae

    else:
        # Create figure
        plt.figure(figsize=(12, 6))
        if prediction_size > 1:
            plt.plot(original.index, maes, label="Stepwise window MAE")
            plt_title = (
                "Accuracy: Multi-step forecast\n Average MAE:"
                f" {mae} with"
                f" prediction_size: {prediction_size} on complete"
                f" test_set: {len(original)}"
            )
            plt_label = "Stepwise window MAE"
        else:
            plt.plot(original.index, original, label="Original data")
            plt.plot(original.index, prediction, label="Stepwise forecast")
            plt_title = (
                f"Accuracy: Single-step forecast\n Average MAE: {mae}"
                f" on complete test_set: {len(original)}"
            )
            plt_label = "BTC in USD"
        plt.title(plt_title)
        plt.xlabel("Datetime")
        plt.ylabel(plt_label)
        plt.legend()
        if save_fig:
            plt.savefig(f"{config.OUTPUT_PATH}/accuracy.png")
        plt.show()


def plot_forecast(
    model,
    df,
    scaler,
    features=config.FEATURES,
    label=config.LABEL,
    window_size=config.WINDOW_SIZE,
    prediction_size=config.PREDICTION_SIZE,
    save_fig=False,
    save_csv=False,
):
    """Predict BTC-USD prices for next days.

    Args:
      model: keras.engine.functional.Functional
        Trained network
      df: pandas.core.frame.DataFrame
        DataFrame from wrangle_data()
      scaler: sklearn.preprocessing._data.MinMaxScaler
        Expect scaler from wrangle_data()
      features: list
        Features used for prediction
      label: string
        Feature to predict
      window_size: integer
        Feature window size for an prediction (Prefix)
      prediction_size: integer
        Size of one prediction
      save_fig: boolean
        True, when plot should be saved
      save_csv: boolean
        True, stores the prediction in a csv file

    Returns:
      plotly.graph_objs._figure.Figure
    """
    DATE_TIME_STRING_FORMAT = "%Y-%m-%d %H:%M:%S"

    # Get last 28 days of data
    data = df[-28:]
    from_date_time = datetime.strptime(
        data.index[-1].strftime(DATE_TIME_STRING_FORMAT),
        DATE_TIME_STRING_FORMAT,
    )
    to_date_time = datetime.strptime(
        (from_date_time + timedelta(days=8)).strftime("%Y-%m-%d 00:00:00"),
        DATE_TIME_STRING_FORMAT,
    )

    # All date times from now until next Sunday
    date_times = [from_date_time.strftime(DATE_TIME_STRING_FORMAT)]
    date_time = from_date_time
    while date_time < to_date_time:
        date_time += timedelta(days=1)
        date_times.append(date_time.strftime(DATE_TIME_STRING_FORMAT))
    date_times = date_times[1:-1]

    if len(features) > 1 and prediction_size < len(date_times):
        print(
            "[INFO] A rolling multivariate single or multi-step"
            + " forecast does currently not exist!"
            + "\n...prediction_size must be greater than:"
            + f" {len(date_times)}"
        )

    else:
        # Univariate or multivariate multi-step model
        if prediction_size >= len(date_times):
            base = df[features][-window_size:]
            forecast = Series(_get_prediction(model, base, include_label=False)[0])
            forecast_times = [
                datetime.strptime(i, DATE_TIME_STRING_FORMAT) for i in date_times
            ]

            if forecast.shape[0] > len(forecast_times):
                forecast = forecast.head(len(forecast_times))
            forecast.index = forecast_times
            forecast = data[label].append(forecast)
            forecast = _rescale_feature(forecast, scaler)

            pred = DataFrame(
                {
                    "forecast": forecast,
                    "strptime": forecast.index,
                    "weekend": [
                        i.strftime("%A").endswith(("Saturday", "Sunday"))
                        for i in forecast.index
                    ],
                    "type": config.WINDOW_SIZE * ["original"]
                    + config.PREDICTION_SIZE * ["forecast"],
                }
            )

        # Univariate single or multi-step model
        elif len(features) == 1 and prediction_size < len(date_times):
            base = [list(df[label][-window_size:])]
            prediction_steps = list(range(0, len(date_times), prediction_size))

            # Forecast BTC-USD prices
            if prediction_size > 1:
                for i in prediction_steps:
                    forecast = model.predict(base[-window_size:])
                    base[0].extend(forecast[0])
                skip = len(base[0][window_size:]) - len(date_times)
                if skip != 0:
                    forecast = Series(
                        base[0][-(len(date_times) + data.shape[0] + skip) :][:-skip]
                    )
                else:
                    forecast = Series(base[0][-(len(date_times) + data.shape[0]) :])

            else:
                for i in prediction_steps:
                    forecast = model.predict(base[-window_size:])
                    base[0].append(forecast[0][0])
                forecast = Series(base[0][-(len(date_times) + data.shape[0]) :])

            # Combine last and predicted BTC-USD prices
            # with date times and a weekend flag
            forecast = _rescale_feature(forecast, scaler)
            forecast_times = [
                datetime.strptime(
                    i.strftime(DATE_TIME_STRING_FORMAT), DATE_TIME_STRING_FORMAT
                )
                for i in data.index
            ]
            forecast_times.extend(
                [datetime.strptime(i, DATE_TIME_STRING_FORMAT) for i in date_times]
            )
            pred = DataFrame(
                {
                    "forecast": forecast,
                    "strptime": forecast_times,
                    "weekend": [
                        i.strftime("%A").endswith(("Saturday", "Sunday"))
                        for i in forecast_times
                    ],
                    "type": config.WINDOW_SIZE * ["original"]
                    + config.PREDICTION_SIZE * ["forecast"],
                }
            )

        # Plot creation
        plt.subplots(figsize=(18, 8))
        if os.path.isfile("assets/forecast.csv"):

            # Compare last forecasts with original values
            last = read_csv("assets/forecast.csv", index_col=0)
            last["strptime"] = to_datetime(last["strptime"])
            last_dates = list(last[last["type"] == "forecast"].index)
            original = pred[
                (pred.index.isin(last_dates)) & (pred["type"] != "forecast")
            ]

            if original.shape[0] != 0:

                # Calcualte the Mean Absolute Error
                original_dates = [i.strftime("%Y-%m-%d") for i in original.index]
                last_forecast = last[last.index.isin(original_dates)]
                mae = round(
                    abs(last_forecast["forecast"].mean() - original["forecast"].mean()),
                    2,
                )
                last_original = last[last["type"] == "original"]

                # Forecast of last week
                plt.subplots_adjust(hspace=0.4)
                plt.subplot(2, 1, 1)
                plt.plot(
                    last_original["strptime"],
                    last_original["forecast"],
                    label="Last Prices",
                )
                plt.plot(
                    [last_original["strptime"][-1]] + list(original["strptime"]),
                    [last_original["forecast"][-1]] + list(original["forecast"]),
                    label="Original Prices",
                    color="tab:blue",
                    linestyle="dotted",
                )
                plt.plot(
                    [last_original["strptime"][-1]] + list(last_forecast["strptime"]),
                    [last_original["forecast"][-1]] + list(last_forecast["forecast"]),
                    label="Predicted Prices",
                )
                plt.ylabel("Bitcoin in USD")
                plt.title(
                    r"$\bf{Validation\ of\ last\ week's\ forecast}$"
                    + f"\nFrom {last_forecast.index[0]} to {last_forecast.index[-1]}: MAE of {mae} USD"
                )
                plt.legend(loc="upper left")

                # Create weekend flags
                ax = plt.gca()
                for idx, weekend in enumerate(last_original["weekend"]):
                    if weekend and idx != (last_original.shape[0] - 1):
                        ax.axvspan(
                            last_original["strptime"][idx],
                            last_original["strptime"][idx + 1],
                            alpha=0.1,
                        )

                # Forecast for next week
                plt.subplot(2, 1, 2)
                plt.plot(
                    pred["strptime"][: data.shape[0]],
                    pred["forecast"][: data.shape[0]],
                    label="Last Prices",
                )
                plt.plot(
                    pred["strptime"][data.shape[0] - 1 :],
                    pred["forecast"][data.shape[0] - 1 :],
                    label="Predicted Prices",
                )

            else:
                plt.plot(
                    pred["strptime"][: data.shape[0]],
                    pred["forecast"][: data.shape[0]],
                    label="Last Prices",
                )
                plt.plot(
                    pred["strptime"][data.shape[0] - 1 :],
                    pred["forecast"][data.shape[0] - 1 :],
                    label="Predicted Prices",
                )

        else:
            plt.plot(
                pred["strptime"][: data.shape[0]],
                pred["forecast"][: data.shape[0]],
                label="Last Prices",
            )
            plt.plot(
                pred["strptime"][data.shape[0] - 1 :],
                pred["forecast"][data.shape[0] - 1 :],
                label="Predicted Prices",
            )

        plt.title(
            r"$\bf{Daily\ Bitcoin\ USD\ closing\ price\ forecast}$"
            + f"\nFrom {date_times[0][:10]} to {date_times[-1][:10]}"
        )
        plt.xlabel("Datetime")
        plt.ylabel("Bitcoin in USD")
        ax = plt.gca()
        for idx, weekend in enumerate(pred["weekend"]):
            if weekend and idx != (pred.shape[0] - 1):
                ax.axvspan(pred["strptime"][idx], pred["strptime"][idx + 1], alpha=0.1)
        plt.legend(loc="upper left")

        if save_csv:
            pred.to_csv(f"{config.OUTPUT_PATH}/forecast.csv")
        if save_fig:
            plt.savefig(f"{config.OUTPUT_PATH}/forecast.png")
        plt.show()
