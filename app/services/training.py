# app/services/training.py

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "newsgroups-classifier"

CATEGORIES = [
    "sci.space",
    "alt.atheism",
    "talk.religion.misc",
    "soc.religion.christian",
    "sci.med",
]


def train_model(
    C: float = 1.0,
    max_features: int = 50000,
    ngram_range: tuple = (1, 1),
    max_iter: int = 1000,
) -> dict:
    """Train the newsgroups text classification pipeline and log
    everything to MLflow — params, metrics, and the model artifact.

    Parameters are explicitly surfaced as function arguments rather
    than hardcoded so every caller can vary them independently and
    MLflow tracks each combination as a separate, comparable run.

    Returns a dict containing the run_id and evaluation metrics so
    the caller can decide whether to register this model or discard it.
    """
    # Pointing MLflow to the tracking server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    train = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
    )
    test = fetch_20newsgroups(
        subset="test",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
    )

    # Build pipeline
    pipeline = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=max_features,
                    ngram_range=ngram_range,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                ),
            ),
        ]
    )

    # Train and evaluate
    pipeline.fit(train.data, train.target)

    training_accuracy = pipeline.score(train.data, train.target)
    test_accuracy = pipeline.score(test.data, test.target)

    # Log everything to MLflow
    with mlflow.start_run() as run:

        # Parameters — what settings produced this model
        mlflow.log_params(
            {
                "C": C,
                "max_features": max_features,
                "ngram_range": str(ngram_range),
                "max_iter": max_iter,
                "categories": ",".join(CATEGORIES),
                "stop_words": "english",
            }
        )

        mlflow.log_metrics(
            {
                "training_accuracy": training_accuracy,
                "test_accuracy": test_accuracy,
            }
        )

        mlflow.log_dict(
            {"target_names": list(train.target_names)},
            "target_names.json",
        )

        # Model artifact — the actual trained pipeline
        # Input/output signatures tell MLflow what shape data goes
        # in and out of this model, so it can validate calls later

        predictions = pipeline.predict(train.data[:5])
        predicted_labels = np.array(train.target_names)[predictions]

        signature = mlflow.models.infer_signature(
            train.data[:5],
            predicted_labels,
        )

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            registered_model_name=EXPERIMENT_NAME,
        )

        run_id = run.info.run_id

    return {
        "run_id": run_id,
        "training_accuracy": round(training_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "params": {
            "C": C,
            "max_features": max_features,
            "ngram_range": ngram_range,
            "max_iter": max_iter,
        },
    }
