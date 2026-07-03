"""
Tests for the FastAPI routes in app.main.

These tests patch every service-layer function main.py calls
(train_model, load_production_model, get_production_model_info,
get_target_names, promote_if_better) so route tests only verify HTTP
status codes, response shapes, and app.state wiring — never real
MLflow behavior, which is already covered by test_serving.py,
test_promotion.py, and test_training.py.

Patches target names as imported INTO app.main, since that's where
the route functions actually look them up.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

CHAMPION_INFO = {
    "model_name": "newsgroups-classifier",
    "version": "10",
    "aliases": ["champion"],
    "run_id": "run-abc123",
    "created_at": 1000,
}
TARGET_NAMES = [
    "alt.atheism",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.religion.misc",
]


@pytest.fixture
def patched_services():
    """Patch every service-layer dependency of app.main with sensible
    defaults representing a healthy, fully-loaded champion model.
    Yields a dict of the mocks so individual tests can override
    return_value / side_effect for specific scenarios.
    """
    fake_model = MagicMock(name="champion_model")
    fake_model.predict_proba.return_value = np.array([[0.05, 0.05, 0.8, 0.05, 0.05]])

    with ExitStack() as stack:
        mocks = {
            "load_production_model": stack.enter_context(
                patch("app.main.load_production_model", return_value=fake_model)
            ),
            "get_production_model_info": stack.enter_context(
                patch(
                    "app.main.get_production_model_info",
                    return_value=dict(CHAMPION_INFO),
                )
            ),
            "get_target_names": stack.enter_context(
                patch("app.main.get_target_names", return_value=list(TARGET_NAMES))
            ),
            "train_model": stack.enter_context(patch("app.main.train_model")),
            "promote_if_better": stack.enter_context(
                patch("app.main.promote_if_better")
            ),
        }
        mocks["fake_model"] = fake_model
        yield mocks


@pytest.fixture
def client(patched_services):
    """A TestClient with lifespan startup running against the patched
    service layer, so app.state gets populated with fake-but-realistic
    values before any route test runs.
    """
    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    def test_health_reports_champion_loaded_true_when_model_present(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["champion_loaded"] is True
        assert body["model_version"] == "10"

    def test_health_reports_false_when_no_champion(self, patched_services):
        patched_services["load_production_model"].return_value = None
        patched_services["get_production_model_info"].return_value = None
        with TestClient(app) as client:
            response = client.get("/health")
        body = response.json()
        assert body["champion_loaded"] is False
        assert body["model_version"] is None


class TestModelInfoEndpoint:
    def test_returns_champion_metadata(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        body = response.json()
        assert body["model_name"] == "newsgroups-classifier"
        assert body["version"] == "10"
        assert body["aliases"] == ["champion"]
        assert body["run_id"] == "run-abc123"

    def test_returns_404_when_no_champion_loaded(self, patched_services):
        patched_services["get_production_model_info"].return_value = None
        with TestClient(app) as client:
            response = client.get("/model/info")
        assert response.status_code == 404


class TestPredictEndpoint:
    def test_returns_category_and_confidence(self, client):
        response = client.post(
            "/predict",
            json={"text": "NASA launched a new spacecraft into orbit today"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["category"] == "sci.space"
        assert body["confidence"] == 0.8
        assert body["model_version"] == "10"

    def test_returns_503_when_no_champion_model(self, patched_services):
        patched_services["load_production_model"].return_value = None
        with TestClient(app) as client:
            response = client.post("/predict", json={"text": "some sample text"})
        assert response.status_code == 503
        assert "No champion model" in response.json()["detail"]

    def test_returns_503_when_target_names_missing(self, patched_services):
        """Regression test for the pre-fix champion version scenario —
        model loaded, but no target_names.json artifact available.
        """
        patched_services["get_target_names"].side_effect = Exception(
            "artifact not found"
        )
        with TestClient(app) as client:
            response = client.post("/predict", json={"text": "some sample text"})
        assert response.status_code == 503
        assert "category metadata" in response.json()["detail"]

    def test_category_maps_to_highest_probability_index(self, client, patched_services):
        patched_services["fake_model"].predict_proba.return_value = np.array(
            [[0.7, 0.1, 0.1, 0.05, 0.05]]
        )
        response = client.post("/predict", json={"text": "atheism discussion"})
        assert response.json()["category"] == "alt.atheism"


class TestTrainEndpoint:
    def test_returns_201_with_run_details(self, client, patched_services):
        patched_services["train_model"].return_value = {
            "run_id": "run-new1",
            "training_accuracy": 0.95,
            "test_accuracy": 0.80,
            "params": {
                "C": 1.0,
                "max_features": 50000,
                "ngram_range": (1, 1),
                "max_iter": 1000,
            },
        }

        response = client.post(
            "/train",
            json={
                "C": 1.0,
                "max_features": 50000,
                "ngram_range": [1, 1],
                "max_iter": 1000,
            },
        )

        assert response.status_code == 201
        body = response.json()
        assert body["run_id"] == "run-new1"
        assert body["test_accuracy"] == 0.80
        assert "champion" in body["message"]

    def test_passes_hyperparameters_through_to_train_model(
        self, client, patched_services
    ):
        patched_services["train_model"].return_value = {
            "run_id": "run-new1",
            "training_accuracy": 0.9,
            "test_accuracy": 0.7,
            "params": {},
        }

        client.post(
            "/train",
            json={
                "C": 2.5,
                "max_features": 1000,
                "ngram_range": [1, 2],
                "max_iter": 500,
            },
        )

        _, kwargs = patched_services["train_model"].call_args
        assert kwargs["C"] == 2.5
        assert kwargs["max_features"] == 1000
        assert kwargs["max_iter"] == 500


class TestReloadEndpoint:
    def test_reloads_and_returns_current_champion(self, client):
        response = client.post("/reload")
        assert response.status_code == 200
        body = response.json()
        assert body["version"] == "10"
        assert "reloaded" in body["message"]

    def test_returns_404_when_reload_finds_no_champion(self, patched_services):
        patched_services["load_production_model"].return_value = None
        with TestClient(app) as client:
            response = client.post("/reload")
        assert response.status_code == 404


class TestPromoteEndpoint:
    def test_returns_promotion_result_when_promoted(self, client, patched_services):
        patched_services["promote_if_better"].return_value = {
            "promoted": True,
            "reason": "Challenger outperformed the current champion.",
            "new_champion_version": "11",
            "challenger_accuracy": 0.9,
            "previous_champion_accuracy": 0.8,
            "previous_champion_version": "10",
        }

        response = client.post("/promote")

        assert response.status_code == 200
        body = response.json()
        assert body["promoted"] is True
        assert body["new_champion_version"] == "11"

    def test_reloads_champion_state_when_promotion_happens(
        self, client, patched_services
    ):
        patched_services["promote_if_better"].return_value = {
            "promoted": True,
            "reason": "promoted",
        }
        call_count_before = patched_services["load_production_model"].call_count

        client.post("/promote")

        # reload_champion() calls load_production_model again after a promotion
        assert patched_services["load_production_model"].call_count > call_count_before

    def test_does_not_reload_when_not_promoted(self, client, patched_services):
        patched_services["promote_if_better"].return_value = {
            "promoted": False,
            "reason": "Challenger did not outperform the current champion.",
        }
        call_count_before = patched_services["load_production_model"].call_count

        response = client.post("/promote")

        assert response.status_code == 200
        assert response.json()["promoted"] is False
        assert patched_services["load_production_model"].call_count == call_count_before
