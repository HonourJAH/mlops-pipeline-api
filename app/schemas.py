from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    C: float = Field(default=1.0, description="Regularization strength")
    max_features: int = Field(default=50000, description="Max vocabulary size")
    ngram_range: tuple = Field(default=(1, 1), description="N-gram range")
    max_iter: int = Field(default=1000, description="Max solver iterations")


class TrainResponse(BaseModel):
    run_id: str
    training_accuracy: float
    test_accuracy: float
    params: dict
    message: str


class PredictRequest(BaseModel):
    text: str = Field(
        min_length=10, description="Text to classify, minimum 10 characters"
    )


class PredictResponse(BaseModel):
    text: str
    category: str
    confidence: float
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    aliases: list[str]
    run_id: str
