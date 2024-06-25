import pytest

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    load_model,
)
from sklearn.linear_model import LogisticRegression

# Define variables for the tests
data = pd.read_csv('data/census.csv')

# categorical variables
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Split the data
train, test = train_test_split(data, test_size=0.2, random_state=0)

# Process the data
X_train, y_train, encoder, lb, scaler = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data
X_test, y_test, _, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb, scaler=scaler
)

# Train the model
model = train_model(X_train, y_train)


def test_train_model():
    """
    Test if the model training function returns a LogisticRegression model.
    """
    assert isinstance(model, LogisticRegression), "The model is not an instance of LogisticRegression"

def test_inference():
    """
    Test if the inference function returns a numpy array.
    """
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray), "Inference did not return a numpy array"

def test_compute_model_metrics():
    """
    Test if the compute_model_metrics function returns the expected number of metrics.
    """
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float), "Precision is not a float"
    assert isinstance(recall, float), "Recall is not a float"
    assert isinstance(fbeta, float), "F1 score is not a float"

def test_data_processing():
    """
    Test if the processed data has the expected type and shape.
    """
    assert isinstance(X_train, np.ndarray), "X_train is not a numpy array"
    assert isinstance(y_train, np.ndarray), "y_train is not a numpy array"
    assert X_train.shape[0] == y_train.shape[0], "The number of samples in X_train and y_train do not match"

def test_save_and_load_model(tmp_path):
    """
    Test if the saved model can be correctly loaded.
    """
    model_path = os.path.join(tmp_path, "model.pkl")
    save_model(model, model_path)
    loaded_model = load_model(model_path)
    assert isinstance(loaded_model, LogisticRegression), "Loaded model is not an instance of LogisticRegression"
# Run the tests
if __name__ == "__main__":
    pytest.main()
