# app/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from app.schemas import (
    TrainRequest,
    TrainResponse,
    PredictRequest,
    PredictResponse,
    ModelInfoResponse,
)
from app.services.training import train_model
from app.services.serving import (
    load_production_model,
    get_production_model_info,
    get_target_names,
)
from app.services.promotion import promote_if_better


def reload_champion(app: FastAPI):
    """Load the current champion model from MLflow into app.state.
    Called at startup, after /reload, and after a successful /promote.
    """
    app.state.champion_model = load_production_model()
    app.state.champion_info = get_production_model_info()
    try:
        app.state.target_names = (
            get_target_names(app.state.champion_info["run_id"])
            if app.state.champion_info
            else None
        )
    except Exception:
        app.state.target_names = None


# Startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    reload_champion(app)
    yield


app = FastAPI(
    title="MLOps Pipeline API",
    description="Train, track, and serve a newsgroups text classifier via MLflow",
    lifespan=lifespan,
)


# Routes
@app.post("/train", status_code=status.HTTP_201_CREATED)
async def train(request: TrainRequest) -> TrainResponse:
    result = train_model(
        C=request.C,
        max_features=request.max_features,
        ngram_range=request.ngram_range,
        max_iter=request.max_iter,
    )

    return TrainResponse(
        run_id=result["run_id"],
        training_accuracy=result["training_accuracy"],
        test_accuracy=result["test_accuracy"],
        params=result["params"],
        message=(
            f"Training complete. Run ID: {result['run_id']}. "
            f"Review metrics in the MLflow UI and promote to @champion "
            f"if this version outperforms the current one."
        ),
    )


@app.post("/predict")
async def predict(request: Request, body: PredictRequest) -> PredictResponse:
    champion_model = request.app.state.champion_model
    champion_info = request.app.state.champion_info
    target_names = request.app.state.target_names

    if champion_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No champion model is currently deployed.",
        )
    if target_names is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Champion model is missing category metadata — retrain and promote a new version.",
        )

    probs = champion_model.predict_proba([body.text])
    class_idx = int(probs.argmax())
    confidence = round(float(probs[0][class_idx]), 4)
    category = target_names[class_idx]

    return PredictResponse(
        text=body.text,
        category=category,
        confidence=confidence,
        model_version=champion_info["version"] if champion_info else "unknown",
    )


@app.post("/reload", status_code=status.HTTP_200_OK)
async def reload_model(request: Request):
    reload_champion(request.app)

    champion_model = request.app.state.champion_model
    champion_info = request.app.state.champion_info

    if champion_model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No champion model found in MLflow registry.",
        )

    return {
        "message": f"Champion model reloaded — now serving version {champion_info['version']}",
        "model_name": champion_info["model_name"],
        "version": champion_info["version"],
        "aliases": champion_info["aliases"],
        "run_id": champion_info["run_id"],
    }


@app.post("/promote")
async def promote(request: Request):
    result = promote_if_better()

    if result["promoted"]:
        reload_champion(request.app)

    return result


@app.get("/model/info")
async def model_info(request: Request):
    champion_info = request.app.state.champion_info

    if champion_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No champion model is currently loaded.",
        )

    return ModelInfoResponse(
        model_name=champion_info["model_name"],
        version=champion_info["version"],
        aliases=champion_info["aliases"],
        run_id=champion_info["run_id"],
    )


@app.get("/health")
async def health_check(request: Request):
    champion_model = request.app.state.champion_model
    champion_info = request.app.state.champion_info

    return {
        "status": "healthy",
        "champion_loaded": champion_model is not None,
        "model_version": champion_info["version"] if champion_info else None,
    }
