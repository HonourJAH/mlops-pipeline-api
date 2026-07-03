"""
Tests for app.services.promotion.promote_if_better().

Every branch of the decision logic gets its own test:
  1. No registered versions at all
  2. No champion yet -> auto-promote latest
  3. Latest version IS already the champion -> no-op
  4. Challenger beats champion -> promote
  5. Challenger ties champion -> do NOT promote (strict > only)
  6. Challenger loses to champion -> do NOT promote

All tests inject a MagicMock client directly (dependency injection) —
no global patching of MlflowClient is needed.
"""

from unittest.mock import patch

from app.services.promotion import promote_if_better, CHAMPION_ALIAS


class TestPromoteIfBetter:
    def test_no_registered_versions_returns_not_promoted(self, mock_client):
        mock_client.search_model_versions.return_value = []

        result = promote_if_better(client=mock_client)

        assert result["promoted"] is False
        assert "No registered model versions" in result["reason"]
        mock_client.set_registered_model_alias.assert_not_called()

    def test_no_existing_champion_auto_promotes_latest(
        self, mock_client, make_version, make_mlflow_run
    ):
        latest = make_version(version="1", run_id="run-1")
        mock_client.search_model_versions.return_value = [latest]
        mock_client.get_run.return_value = make_mlflow_run(
            run_id="run-1", test_accuracy=0.70
        )
        mock_client.get_model_version_by_alias.side_effect = Exception(
            "no champion yet"
        )

        result = promote_if_better(client=mock_client)

        assert result["promoted"] is True
        assert result["new_champion_version"] == "1"
        assert result["previous_champion_accuracy"] is None
        mock_client.set_registered_model_alias.assert_called_once_with(
            name="newsgroups-classifier",
            alias=CHAMPION_ALIAS,
            version="1",
        )

    def test_latest_version_already_champion_is_a_noop(
        self, mock_client, make_version, make_mlflow_run
    ):
        current = make_version(version="5", run_id="run-5")
        mock_client.search_model_versions.return_value = [current]
        mock_client.get_run.return_value = make_mlflow_run(
            run_id="run-5", test_accuracy=0.80
        )
        mock_client.get_model_version_by_alias.return_value = current

        result = promote_if_better(client=mock_client)

        assert result["promoted"] is False
        assert "already the champion" in result["reason"]
        mock_client.set_registered_model_alias.assert_not_called()

    def test_challenger_outperforms_champion_gets_promoted(
        self, mock_client, make_version, make_mlflow_run
    ):
        challenger = make_version(version="2", run_id="run-2")
        champion = make_version(version="1", run_id="run-1")

        mock_client.search_model_versions.return_value = [challenger]
        mock_client.get_model_version_by_alias.return_value = champion

        def fake_get_run(run_id):
            accuracies = {"run-1": 0.70, "run-2": 0.85}
            return make_mlflow_run(run_id=run_id, test_accuracy=accuracies[run_id])

        mock_client.get_run.side_effect = fake_get_run

        result = promote_if_better(client=mock_client)

        assert result["promoted"] is True
        assert result["new_champion_version"] == "2"
        assert result["challenger_accuracy"] == 0.85
        assert result["previous_champion_accuracy"] == 0.70
        assert result["previous_champion_version"] == "1"
        mock_client.set_registered_model_alias.assert_called_once_with(
            name="newsgroups-classifier",
            alias=CHAMPION_ALIAS,
            version="2",
        )

    def test_challenger_losing_to_champion_does_not_promote(
        self, mock_client, make_version, make_mlflow_run
    ):
        challenger = make_version(version="2", run_id="run-2")
        champion = make_version(version="1", run_id="run-1")

        mock_client.search_model_versions.return_value = [challenger]
        mock_client.get_model_version_by_alias.return_value = champion

        def fake_get_run(run_id):
            accuracies = {"run-1": 0.90, "run-2": 0.60}
            return make_mlflow_run(run_id=run_id, test_accuracy=accuracies[run_id])

        mock_client.get_run.side_effect = fake_get_run

        result = promote_if_better(client=mock_client)

        assert result["promoted"] is False
        assert result["champion_version"] == "1"
        assert result["challenger_version"] == "2"
        mock_client.set_registered_model_alias.assert_not_called()

    def test_tie_does_not_promote(self, mock_client, make_version, make_mlflow_run):
        """Equal accuracy is a deliberate design decision: ties keep the
        existing champion rather than promoting (strict > comparison).
        """
        challenger = make_version(version="2", run_id="run-2")
        champion = make_version(version="1", run_id="run-1")

        mock_client.search_model_versions.return_value = [challenger]
        mock_client.get_model_version_by_alias.return_value = champion

        def fake_get_run(run_id):
            return make_mlflow_run(run_id=run_id, test_accuracy=0.75)

        mock_client.get_run.side_effect = fake_get_run

        result = promote_if_better(client=mock_client)

        assert result["promoted"] is False
        mock_client.set_registered_model_alias.assert_not_called()

    def test_picks_most_recently_created_version_as_latest(
        self, mock_client, make_version, make_mlflow_run
    ):
        """search_model_versions is expected to be called with
        order_by=creation_timestamp DESC, so the function should use
        result[0] as 'latest' without additional sorting of its own.
        """
        newest = make_version(version="9", run_id="run-9")
        mock_client.search_model_versions.return_value = [newest]
        mock_client.get_model_version_by_alias.side_effect = Exception("none yet")
        mock_client.get_run.return_value = make_mlflow_run(
            run_id="run-9", test_accuracy=0.5
        )

        result = promote_if_better(client=mock_client)

        assert result["new_champion_version"] == "9"
        _, kwargs = mock_client.search_model_versions.call_args
        assert "creation_timestamp DESC" in kwargs.get("order_by", [""])[0]

    def test_creates_default_client_and_sets_tracking_uri_when_none_provided(
        self, make_version, make_mlflow_run
    ):
        with patch("app.services.promotion.MlflowClient") as client_cls, patch(
            "app.services.promotion.mlflow.set_tracking_uri"
        ) as set_uri:
            fake_client = client_cls.return_value
            fake_client.search_model_versions.return_value = [
                make_version(version="1", run_id="run-1")
            ]
            fake_client.get_model_version_by_alias.side_effect = Exception("none")
            fake_client.get_run.return_value = make_mlflow_run(
                run_id="run-1", test_accuracy=0.5
            )

            promote_if_better()

            set_uri.assert_called_once()
            client_cls.assert_called_once()
