"""
Shared fixtures for the MLOps pipeline API test suite.

These fixtures fabricate lightweight stand-ins for MLflow's real return
objects (ModelVersion, Run) so tests never need a live MLflow tracking
server, a real model registry, or real training data.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def make_model_version(
    version="1",
    run_id="run-abc123",
    aliases=None,
    name="newsgroups-classifier",
    creation_timestamp=1000,
):
    """Fabricate an object shaped like mlflow.entities.model_registry.ModelVersion.

    Real ModelVersion.aliases is not a plain list under the hood, which is
    exactly the bug that broke jsonable_encoder earlier in this project —
    so tests exercise this using a plain list, matching what our own code
    is responsible for casting it to (list(version.aliases)) before it
    reaches serving.py's return value.
    """
    return SimpleNamespace(
        name=name,
        version=version,
        run_id=run_id,
        aliases=aliases if aliases is not None else ["champion"],
        creation_timestamp=creation_timestamp,
    )


def make_run(run_id="run-abc123", test_accuracy=0.85, training_accuracy=0.95):
    """Fabricate an object shaped like mlflow.entities.Run."""
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id),
        data=SimpleNamespace(
            metrics={
                "test_accuracy": test_accuracy,
                "training_accuracy": training_accuracy,
            },
            params={},
        ),
    )


@pytest.fixture
def make_version():
    """Factory fixture so individual tests can build custom versions."""
    return make_model_version


@pytest.fixture
def make_mlflow_run():
    """Factory fixture so individual tests can build custom runs."""
    return make_run


@pytest.fixture
def mock_client():
    """A bare MagicMock standing in for mlflow.tracking.MlflowClient.

    Individual tests configure the specific methods they need
    (get_model_version_by_alias, search_model_versions, get_run,
    set_registered_model_alias, download_artifacts) via return_value
    or side_effect.
    """
    return MagicMock()
