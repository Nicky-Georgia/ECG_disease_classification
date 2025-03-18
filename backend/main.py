from fastapi import FastAPI
from app.schemas import (
    ModelListResponse, SetActiveModelRequest, SetActiveModelResponse,
    FitParams, FitResponse, PredictRequest, PredictResponse,
    FineTuneResponse
)
from app.trainer import train_model_async
from app.managers.model_manager import model_manager
from app.utils.logger import logger
from app.utils.data_loader import X_train, y_train, X_test, y_test

app = FastAPI(title="ML Model Service")

@app.on_event("startup")
async def startup_event():
    logger.info("Server starting up...")
    model_manager.load_models()

@app.get("/models", response_model=ModelListResponse)
async def list_models():
    models = model_manager.list_models()
    return ModelListResponse(models=models)

@app.post("/set", response_model=SetActiveModelResponse)
async def set_active_model(request: SetActiveModelRequest):
    success = model_manager.set_active_model(request.model_id)
    if not success:
        return SetActiveModelResponse(status="error", detail=f"Model {request.model_id} not found")
    return SetActiveModelResponse(status="success", detail=f"Active model set to {request.model_id}")

@app.post("/fit", response_model=FitResponse)
async def fit_model(params: FitParams):
    active_model = model_manager.get_active_model()
    if not active_model:
        return FitResponse(status="error", detail="No active model to train.")
    train_model_async(active_model, X_train, y_train, params.dict())
    return FitResponse(status="success", detail=f"Training started for model {active_model.id}", model_id=active_model.id)

@app.post("/fine_tune", response_model=FineTuneResponse)
async def fine_tune_model(params: FitParams):
    active_model = model_manager.get_active_model()
    if not active_model:
        return FineTuneResponse(status="error", detail="No active model to fine-tune.")
    train_model_async(active_model, X_train, y_train, params.dict())
    return FineTuneResponse(status="success", detail=f"Fine-tune started for model {active_model.id}", model_id=active_model.id)

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    active_model = model_manager.get_active_model()
    if not active_model:
        return PredictResponse(predictions=[])
    predictions = active_model.predict(request.data)
    return PredictResponse(predictions=predictions)
