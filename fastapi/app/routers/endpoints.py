from fastapi import APIRouter, HTTPException, UploadFile, File
from app.schemas.schemas import (
    PredictRequest, PredictResponse,
    FitRequest, FitResponse,
    ModelsListResponse, SetModelRequest, SetModelResponse,
    UploadResponse, CreateModelRequest
)
from app.managers.model_manager import model_manager
from app.trainer import train_model
from app.data.loader import load_heartbeat_data
from app.utils.logger import logger
import numpy as np
import joblib
import os
from enum import Enum
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import List, Any, Dict, Optional

router = APIRouter()


class ArchitectureEnum(str, Enum):
    decision_tree = "DecisionTree"
    random_forest = "RandomForest"
    gradient_boosting = "GradientBoosting"

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

    # Получаем активную модель
    model = model_manager.get_active_model()
    if not model:
        raise HTTPException(status_code=400, detail="No active model to train")

    # Проверяем, была ли модель уже обучена
    if not isinstance(model, RandomForestClassifier) or not hasattr(model, "estimators_"):
        logger.info("Model is not trained yet. Proceeding with training.")
    else:
        logger.info("Model already trained.")

    try:
        # Загружаем тренировочные данные
        X_train, y_train, X_val, y_val, X_test, y_test = load_heartbeat_data()
        logger.info(f"Loaded data with shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")

        # Обучаем модель с переданными гиперпараметрами
        train_model(model, X_train, y_train, request.params)

        return FitResponse(message="Model training started. Timeout: 10 seconds")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    logger.info("Models list endpoint called")
    models = model_manager.get_models_info()
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
        temp_file_path = f"app/models/{file.filename}"

        #os.makedirs("./models", exist_ok=True)
        with open(temp_file_path, "wb") as f:
            f.write(contents)

        model_object = joblib.load(temp_file_path)
        model_id = os.path.splitext(file.filename)[0]

        model_manager.add_model(model_id, model_object)

        return UploadResponse(message=f"Model {model_id} uploaded and ready")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def pre_fit(model):
    """
    pre-обучение модели на фиктивных данных,
    чтобы проинициализировать внутренние атрибуты.
    """
    X_dummy = np.array([[0, 0], [1, 1]])
    y_dummy = np.array([0, 1])           

    try:
        model.fit(X_dummy, y_dummy)
        logger.info(f"Pre fit completed for model: {type(model).__name__}")
    except Exception as e:
        logger.error(f"Pre fit failed for model {type(model).__name__}: {e}")
        raise e

    return model


def build_model(architecture: ArchitectureEnum, params: Optional[Dict] = None): 
    params = params or {}

    if architecture == ArchitectureEnum.decision_tree:
        model = DecisionTreeClassifier(**params)
    elif architecture == ArchitectureEnum.random_forest:
        model = RandomForestClassifier(**params)
    elif architecture == ArchitectureEnum.gradient_boosting:
        model = GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    if hasattr(model, 'fit'):
        logger.info(f"Model {architecture} successfully initialized.")
    else:
        raise ValueError(f"Failed to initialize model: {architecture}")

    pre_fit(model)
    
    return model


@router.post("/create_model", response_model=SetModelResponse)
async def create_model(request: CreateModelRequest):
    logger.info(f"Creating model '{request.model_id}' with architecture '{request.architecture}'")

    # Проверяем, существует ли уже модель с таким id
    if request.model_id in model_manager.models:
        raise HTTPException(status_code=400, detail=f"Model with id '{request.model_id}' already exists.")

    try:
        # Создаем модель с параметрами
        model = build_model(request.architecture, request.params)

        # Добавляем в менеджер и делаем активной
        model_manager.add_model(request.model_id, model)
        model_manager.set_active_model(request.model_id)

        logger.info(f"Model '{request.model_id}' ({request.architecture}) successfully created and set as active.")
        return SetModelResponse(message=f"Model '{request.model_id}' ({request.architecture}) created and set active.")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

