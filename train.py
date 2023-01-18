"""
Training model to score readability of provided text, using Universal Sentence Encoder embeddings and RandomForestRegressor

Usage:
    python3 training_model.py \
        --input_file "data/train.csv" \
        --model_output_file "output/regressor.pkl" \
        --metrics_output_file "output/metrics.txt"

    Parameters info:
        input_file has .csv extension and has columns "excerpt" (for given text) and "target" (for target score)
"""

import ssl
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import logging
import argparse
import os

ssl._create_default_https_context = ssl._create_unverified_context
MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
RANDOM_STATE = 100
TEXT_COL, TARGET_COL, EMBEDDING_COL = 'excerpt', 'target', 'embedding'


def load_model():
    logging.info("Downloading encoder...")
    model = hub.load(MODEL_URL)
    logging.info("Encoder loaded!")
    return model


def get_text_embeddings(texts, encoder):
    return encoder(texts).numpy()


def train_regressor(data, save_to=None, metrics_to=None):
    X_train, X_test, y_train, y_test = train_test_split(data[EMBEDDING_COL].values, data[TARGET_COL].values,
                                                        test_size=0.3, random_state=RANDOM_STATE)
    train_matrix = np.array([np.array(x) for x in X_train])
    test_matrix = np.array([np.array(x) for x in X_test])
    regressor = RandomForestRegressor(n_estimators=10, random_state=RANDOM_STATE)
    logging.info("Fitting regressor...")
    regressor.fit(train_matrix, y_train)

    if save_to:
        _save_model(regressor, save_to)

    if metrics_to:
        predictions = regressor.predict(test_matrix)
        mse = mean_squared_error(y_test, predictions)
        content = f"Mean Squared Error: {mse}"
        _save_metrics(content, metrics_to)

    return regressor


def _save_model(regressor, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as out_file:
        pickle.dump(regressor, out_file)


def _save_metrics(content, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as out_file:
        out_file.write(content)


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description='Training model for text readability evaluation.')
    parser.add_argument('-i', '--input_file', help='Path to training data in csv format.', required=True)
    parser.add_argument('-r', '--model_output_file', help='Save model to a provided .pkl file.', required=False,
                        default=None)
    parser.add_argument('-m', '--metrics_output_file', help='Save metrics to a provided .txt file.', required=False,
                        default=None)
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    train_data_path = args.input_file
    save_model_file = args.model_output_file
    save_metrics_file = args.metrics_output_file
    encoder = load_model()
    train_data = pd.read_csv(train_data_path)
    train_data[EMBEDDING_COL] = get_text_embeddings(train_data[TEXT_COL].values, encoder).tolist()
    regressor = train_regressor(train_data, save_to=save_model_file, metrics_to=save_metrics_file)
    logging.info("Done!")
