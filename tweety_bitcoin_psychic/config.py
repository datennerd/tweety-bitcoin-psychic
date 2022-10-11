"""Main configuration file."""

# Utils
OUTPUT_PATH = "data"
TRAIN_VAL_TEST_SPLIT = [0.90, 0.95]
TRAIN_VAL_SPLIT = 0.95
SUBSET_SIZE = 0.2

# Window dataset creation
FEATURES = ["close", "high_close_pdiff", "low_close_pdiff", "volume_pchg"]
LABEL = "close"
WINDOW_SIZE = 28
PREDICTION_SIZE = 7
PREDICTION_WINDOW = 7
BATCH_SIZE = 64

# Automated hyperparameter tuning
KERAS_TUNER = "BayesianOptimization"
MAX_LAYERS = 4
TRIALS_FOR_TUNER = 10
EPOCHS_FOR_TUNER = 50
EPOCHS_FOR_TRAINING = 1028
PATIENCE = 32
MAX_LR_FACTOR = 2
