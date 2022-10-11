# Tweety Bitcoin Psychic

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub Super-Linter](https://github.com/datennerd/tweety-bitcoin-psychic/workflows/Lint%20Code%20Base/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub watchers](https://img.shields.io/github/watchers/datennerd/tweety-bitcoin-psychic?style=social)](https://github.com/datennerd/tweety-bitcoin-psychic)
[![GitHub forks](https://img.shields.io/github/forks/datennerd/tweety-bitcoin-psychic?style=social)](https://github.com/datennerd/tweety-bitcoin-psychic)
[![GitHub Repo stars](https://img.shields.io/github/stars/datennerd/tweety-bitcoin-psychic?style=social)](https://github.com/datennerd/tweety-bitcoin-psychic)

This bot is based on a neural network *(LSTM aka. Long Short Term Memory)* that automatically retrains itself every Monday morning and tries to predict the daily Bitcoin-USD closing prices for the entire week. The current forecast is tweeted with the evaluation of last week's forecast at [@BTCPsychic](https://twitter.com/BTCPsychic).

> **Legal Disclaimer**<br/>
> This project is not investment advice, it is intended only to research and educational purposes.

![Banner](banner.png)

## How the code works

The code is written in Python and intended to run as Machine Learning Pipeline *(Data Extraction, Model Training, Model Evaluation, Model Deployment)* on [GitHub Actions](https://docs.github.com/en/actions).
At the start of the job, historical day-based market data is loaded with [yfinance](https://pypi.org/project/yfinance/) and then pre-processed.
Then, the neural network architecture is defined using [TensorFlow](https://www.tensorflow.org) and [KerasTuner](https://keras.io/keras_tuner/) automatically tries different LSTM networks and their hyperparameters.
The network with the "best" MAE (Mean Average Error) is then selected for training.
For this purpose, a Keras callback with an implementation of [Cyclical Learning Rate](https://github.com/bckenstler/CLR) policies is used.
After the current network has been trained, it is compared to the previous week's network.
The best network is then trained again with more data and then used to predict the Bitcoin USD closing price for the next seven days.
And finally, the current forecast and an evaluation of last week's forecast is tweeted using [tweepy](https://github.com/tweepy/tweepy).
