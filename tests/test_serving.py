"""
Tests for app.services.serving — load_production_model(),
get_production_model_info(), get_target_names().

All three functions accept an optional `client` parameter (dependency
injection), so every test here builds a MagicMock client and passes it
in directly rather than patching mlflow.tracking.MlflowClient globally.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.services.serving import (
    load_production_model,
    get_production_model_info,
    get_target_names,
    CHAMPION_ALIAS,
)


class TestLoadProductionModel:
    def test_returns_none_when_no_champion_alias_exists(self, mock_client):
        mock_client.get_model_version_by_alias.side_effect = Exception(
            "RESOURCE_DOES_NOT_EXIST"
        )
        result = load_production_model(client=mock_client)
        assert result is None

    def test_loads_model_when_champion_alias_exists(self, mock_client, make_version):
        mock_client.get_model_version_by_alias.return_value = make_version()

        with patch("app.services.serving.mlflow.sklearn.load_model") as load_model:
            load_model.return_value = "fake-pipeline"
            result = load_production_model(client=mock_client)

        assert result == "fake-pipeline"

    def test_loads_model_using_champion_alias_uri(self, mock_client, make_version):
        mock_client.get_model_version_by_alias.return_value = make_version(
            name="newsgroups-classifier"
        )

        with patch("app.services.serving.mlflow.sklearn.load_model") as load_model:
            load_production_model(client=mock_client)

        called_uri = load_model.call_args[0][0]
        assert called_uri == f"models:/newsgroups-classifier@{CHAMPION_ALIAS}"

    def test_creates_default_client_when_none_provided(self, make_version):
        """Confirms the DI default path still works when no client is passed —
        this is what reload_champion() in main.py relies on.
        """
        with patch("app.services.serving.MlflowClient") as client_cls, patch(
            "app.services.serving.mlflow.set_tracking_uri"
        ) as set_uri, patch("app.services.serving.mlflow.sklearn.load_model"):
            client_cls.return_value.get_model_version_by_alias.return_value = (
                make_version()
            )
            load_production_model()
            set_uri.assert_called_once()
            client_cls.assert_called_once()


class TestGetProductionModelInfo:
    def test_returns_none_when_no_champion(self, mock_client):
        mock_client.get_model_version_by_alias.side_effect = Exception("not found")
        result = get_production_model_info(client=mock_client)
        assert result is None

    def test_returns_metadata_dict_when_champion_exists(
        self, mock_client, make_version
    ):
        mock_client.get_model_version_by_alias.return_value = make_version(
            version="7",
            run_id="run-777",
            aliases=["champion"],
        )

        result = get_production_model_info(client=mock_client)

        assert result == {
            "model_name": "newsgroups-classifier",
            "version": "7",
            "aliases": ["champion"],
            "run_id": "run-777",
            "created_at": 1000,
        }

    def test_aliases_is_cast_to_plain_list(self, mock_client, make_version):
        """Regression test: MLflow's real ModelVersion.aliases is not a plain
        list under the hood, which previously broke FastAPI's jsonable_encoder
        with a 'dictionary update sequence' error. This confirms the returned
        value is always a genuine list, never left as whatever MLflow's
        internal type is.
        """

        class WeirdAliasContainer:
            """Stand-in for MLflow's internal non-list alias container."""

            def __init__(self, items):
                self._items = items

            def __iter__(self):
                return iter(self._items)

        mock_client.get_model_version_by_alias.return_value = make_version(
            aliases=WeirdAliasContainer(["champion"])
        )

        result = get_production_model_info(client=mock_client)

        assert type(result["aliases"]) is list
        assert result["aliases"] == ["champion"]

    def test_version_field_is_returned_as_given(self, mock_client, make_version):
        mock_client.get_model_version_by_alias.return_value = make_version(
            version="10"
        )
        result = get_production_model_info(client=mock_client)
        assert result["version"] == "10"


class TestGetTargetNames:
    def test_downloads_and_parses_target_names_json(
        self, mock_client, tmp_path
    ):
        artifact_file = tmp_path / "target_names.json"
        artifact_file.write_text(
            json.dumps({"target_names": ["sci.space", "sci.med"]})
        )
        mock_client.download_artifacts.return_value = str(artifact_file)

        result = get_target_names("run-abc123", client=mock_client)

        assert result == ["sci.space", "sci.med"]

    def test_requests_correct_artifact_path(self, mock_client, tmp_path):
        artifact_file = tmp_path / "target_names.json"
        artifact_file.write_text(json.dumps({"target_names": []}))
        mock_client.download_artifacts.return_value = str(artifact_file)

        get_target_names("run-abc123", client=mock_client)

        mock_client.download_artifacts.assert_called_once_with(
            "run-abc123", "target_names.json"
        )

    def test_raises_when_artifact_missing(self, mock_client):
        """When a champion version predates the target_names.json fix,
        download_artifacts should raise — callers (main.py's
        reload_champion) are responsible for catching this, not this
        function itself.
        """
        mock_client.download_artifacts.side_effect = Exception(
            "Failed to download artifacts from path 'target_names.json'"
        )

        with pytest.raises(Exception):
            get_target_names("run-old-version", client=mock_client)

    def test_creates_default_client_and_sets_tracking_uri_when_none_provided(
        self, tmp_path
    ):
        artifact_file = tmp_path / "target_names.json"
        artifact_file.write_text(json.dumps({"target_names": ["sci.space"]}))

        with patch("app.services.serving.MlflowClient") as client_cls, patch(
            "app.services.serving.mlflow.set_tracking_uri"
        ) as set_uri:
            client_cls.return_value.download_artifacts.return_value = str(
                artifact_file
            )
            get_target_names("run-abc123")
            set_uri.assert_called_once()
            client_cls.assert_called_once()
