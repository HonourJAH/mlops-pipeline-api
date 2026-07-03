import mlflow
from mlflow.tracking import MlflowClient

from app.services.training import MLFLOW_TRACKING_URI, EXPERIMENT_NAME

CHAMPION_ALIAS = "champion"


def _get_client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def _get_test_accuracy(client: MlflowClient, run_id: str) -> float:
    run = client.get_run(run_id)
    return run.data.metrics["test_accuracy"]


def promote_if_better(client: MlflowClient | None = None) -> dict:
    client = client or _get_client()

    latest_versions = client.search_model_versions(
        f"name='{EXPERIMENT_NAME}'",
        order_by=["creation_timestamp DESC"],
    )
    if not latest_versions:
        return {"promoted": False, "reason": "No registered model versions found."}

    latest_version = latest_versions[0]
    challenger_accuracy = _get_test_accuracy(client, latest_version.run_id)

    try:
        champion_version = client.get_model_version_by_alias(
            name=EXPERIMENT_NAME,
            alias=CHAMPION_ALIAS,
        )
    except Exception:
        champion_version = None

    if champion_version is None:
        client.set_registered_model_alias(
            name=EXPERIMENT_NAME,
            alias=CHAMPION_ALIAS,
            version=latest_version.version,
        )
        return {
            "promoted": True,
            "reason": "No existing champion — promoted automatically.",
            "new_champion_version": latest_version.version,
            "challenger_accuracy": challenger_accuracy,
            "previous_champion_accuracy": None,
        }

    if champion_version.version == latest_version.version:
        return {
            "promoted": False,
            "reason": "Latest version is already the champion.",
            "champion_version": champion_version.version,
            "champion_accuracy": challenger_accuracy,
        }

    champion_accuracy = _get_test_accuracy(client, champion_version.run_id)

    if challenger_accuracy > champion_accuracy:
        client.set_registered_model_alias(
            name=EXPERIMENT_NAME,
            alias=CHAMPION_ALIAS,
            version=latest_version.version,
        )
        return {
            "promoted": True,
            "reason": "Challenger outperformed the current champion.",
            "new_champion_version": latest_version.version,
            "challenger_accuracy": challenger_accuracy,
            "previous_champion_accuracy": champion_accuracy,
            "previous_champion_version": champion_version.version,
        }

    return {
        "promoted": False,
        "reason": "Challenger did not outperform the current champion.",
        "challenger_version": latest_version.version,
        "challenger_accuracy": challenger_accuracy,
        "champion_version": champion_version.version,
        "champion_accuracy": champion_accuracy,
    }
