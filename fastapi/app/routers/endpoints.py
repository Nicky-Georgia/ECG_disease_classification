from fastapi import APIRouter, HTTPException, UploadFile, File
from app.schemas.schemas import (
    PredictRequest, PredictResponse,
    FitRequest, FitResponse,
    ModelsListResponse, SetModelRequest, SetModelResponse,
    UploadResponse
)
from app.managers.model_manager import model_manager
from app.trainer import train_model_async
from app.data.loader import load_heartbeat_data
from app.utils.logger import logger
import numpy as np
import joblib
import os

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    logger.info("Predict endpoint called")

    model = model_manager.get_active_model()
    if not model:
        raise HTTPException(status_code=400, detail="No active model available")

    try:
        X = np.array(request.data).reshape(1, -1)
        prediction = model.predict(X).tolist()
        logger.info(f"Prediction successful: {prediction}")
        return PredictResponse(prediction=prediction)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fit", response_model=FitResponse)
async def fit(request: FitRequest):
    logger.info(f"Fit endpoint called with params: {request.params}")

    model = model_manager.get_active_model()
    if not model:
        raise HTTPException(status_code=400, detail="No active model to train")

    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_heartbeat_data()
        train_model_async(model, X_train, y_train, request.params)

        return FitResponse(message=f"Model training started. Timeout: 10 seconds")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    logger.info("Models list endpoint called")
    models = model_manager.get_models_list()
    return ModelsListResponse(models=models)

@router.post("/set", response_model=SetModelResponse)
async def set_model(request: SetModelRequest):
    logger.info(f"Set model endpoint called with model_id: {request.model_id}")

    success = model_manager.set_active_model(request.model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")

    return SetModelResponse(message=f"Active model switched to {request.model_id}")

@router.post("/upload", response_model=UploadResponse)
async def upload_model(file: UploadFile = File(...)):
    logger.info(f"Upload model endpoint called: {file.filename}")

    try:
        contents = await file.read()
        temp_file_path = f"./models/{file.filename}"

        os.makedirs("./models", exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(contents)

        model_object = joblib.load(temp_file_path)
        model_id = os.path.splitext(file.filename)[0]

        model_manager.add_model(model_id, model_object)

        return UploadResponse(message=f"Model {model_id} uploaded and ready")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))