"""
Tests for app.services.training.train_model().

Two external dependencies are mocked in every test here:
  - sklearn.datasets.fetch_20newsgroups  (avoids a real network download)
  - mlflow                               (avoids a real tracking server)

We patch these at the location they're imported INTO (app.services.training),
not at their original definition site, since that's where train_model()
actually looks them up.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.training import train_model, CATEGORIES


def _fake_newsgroups_bunch(n=20, n_classes=5):
    """Build a minimal object shaped like sklearn's Bunch return value
    from fetch_20newsgroups — just the attributes train_model() touches.
    """
    rng = np.random.default_rng(seed=0)
    return MagicMock(
        data=[f"sample document {i}" for i in range(n)],
        target=rng.integers(0, n_classes, size=n),
        target_names=[
            "alt.atheism",
            "sci.med",
            "sci.space",
            "soc.religion.christian",
            "talk.religion.misc",
        ],
    )


@pytest.fixture
def mock_mlflow(monkeypatch):
    """Patch the mlflow module as imported inside app.services.training."""
    fake_mlflow = MagicMock()
    fake_run_context = MagicMock()
    fake_run_context.__enter__.return_value = MagicMock(
        info=MagicMock(run_id="run-xyz789")
    )
    fake_mlflow.start_run.return_value = fake_run_context
    fake_mlflow.models.infer_signature.return_value = "fake-signature"

    monkeypatch.setattr("app.services.training.mlflow", fake_mlflow)
    return fake_mlflow


@pytest.fixture
def mock_fetch(monkeypatch):
    train_bunch = _fake_newsgroups_bunch(n=40)
    test_bunch = _fake_newsgroups_bunch(n=20)

    def _fetch(subset, categories, remove):
        return train_bunch if subset == "train" else test_bunch

    monkeypatch.setattr(
        "app.services.training.fetch_20newsgroups",
        MagicMock(side_effect=_fetch),
    )
    return train_bunch, test_bunch


class TestTrainModel:
    def test_returns_run_id_from_mlflow_run(self, mock_mlflow, mock_fetch):
        result = train_model()
        assert result["run_id"] == "run-xyz789"

    def test_returns_rounded_accuracy_metrics(self, mock_mlflow, mock_fetch):
        result = train_model()
        assert "training_accuracy" in result
        assert "test_accuracy" in result
        assert 0.0 <= result["training_accuracy"] <= 1.0
        assert 0.0 <= result["test_accuracy"] <= 1.0

    def test_echoes_back_hyperparameters(self, mock_mlflow, mock_fetch):
        result = train_model(C=2.5, max_features=1000, ngram_range=(1, 2), max_iter=500)
        assert result["params"] == {
            "C": 2.5,
            "max_features": 1000,
            "ngram_range": (1, 2),
            "max_iter": 500,
        }

    def test_logs_params_metrics_and_model_to_mlflow(self, mock_mlflow, mock_fetch):
        train_model(C=1.0)
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_metrics.assert_called_once()
        mock_mlflow.sklearn.log_model.assert_called_once()

    def test_logs_target_names_artifact(self, mock_mlflow, mock_fetch):
        """Regression test for the missing target_names.json bug that broke
        /predict earlier in this project — every training run must log the
        class-name order so serving.py can map integer labels back to
        category strings.
        """
        train_model()
        mock_mlflow.log_dict.assert_called_once()
        logged_payload, logged_path = mock_mlflow.log_dict.call_args[0]
        assert logged_path == "target_names.json"
        assert "target_names" in logged_payload
        assert logged_payload["target_names"] == [
            "alt.atheism",
            "sci.med",
            "sci.space",
            "soc.religion.christian",
            "talk.religion.misc",
        ]

    def test_registers_model_under_experiment_name(self, mock_mlflow, mock_fetch):
        from app.services.training import EXPERIMENT_NAME

        train_model()
        _, kwargs = mock_mlflow.sklearn.log_model.call_args
        assert kwargs["registered_model_name"] == EXPERIMENT_NAME

    def test_sets_tracking_uri_and_experiment(self, mock_mlflow, mock_fetch):
        from app.services.training import MLFLOW_TRACKING_URI, EXPERIMENT_NAME

        train_model()
        mock_mlflow.set_tracking_uri.assert_called_once_with(MLFLOW_TRACKING_URI)
        mock_mlflow.set_experiment.assert_called_once_with(EXPERIMENT_NAME)

    def test_uses_default_hyperparameters_when_not_specified(
        self, mock_mlflow, mock_fetch
    ):
        result = train_model()
        assert result["params"]["C"] == 1.0
        assert result["params"]["max_features"] == 50000
        assert result["params"]["ngram_range"] == (1, 1)
        assert result["params"]["max_iter"] == 1000

    def test_fetches_only_configured_categories(self, mock_mlflow, mock_fetch):
        with patch(
            "app.services.training.fetch_20newsgroups"
        ) as fetch_mock:
            fetch_mock.side_effect = lambda subset, categories, remove: (
                _fake_newsgroups_bunch()
            )
            train_model()
            for call in fetch_mock.call_args_list:
                assert call.kwargs["categories"] == CATEGORIES
