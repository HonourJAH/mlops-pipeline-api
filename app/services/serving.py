import json
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from app.services.training import MLFLOW_TRACKING_URI, EXPERIMENT_NAME

CHAMPION_ALIAS = "champion"


def _get_client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def load_production_model(client: MlflowClient | None = None):
    client = client or _get_client()

    try:
        client.get_model_version_by_alias(
            name=EXPERIMENT_NAME,
            alias=CHAMPION_ALIAS,
        )
    except Exception:
        return None

    model_uri = f"models:/{EXPERIMENT_NAME}@{CHAMPION_ALIAS}"
    return mlflow.sklearn.load_model(model_uri)


def get_production_model_info(client: MlflowClient | None = None) -> dict | None:
    client = client or _get_client()

    try:
        version = client.get_model_version_by_alias(
            name=EXPERIMENT_NAME,
            alias=CHAMPION_ALIAS,
        )
    except Exception:
        return None

    return {
        "model_name": version.name,
        "version": version.version,
        "aliases": list(version.aliases),
        "run_id": version.run_id,
        "created_at": version.creation_timestamp,
    }


def get_target_names(run_id: str, client: MlflowClient | None = None) -> list[str]:
    client = client or _get_client()
    local_path = client.download_artifacts(run_id, "target_names.json")
    with open(local_path) as f:
        return json.load(f)["target_names"]
