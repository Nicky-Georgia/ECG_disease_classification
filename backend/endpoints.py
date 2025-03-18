from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.schemas import (
    FitParams, FitResponse, PredictRequest, PredictResponse,
    ModelsListResponse, ModelInfo, SetModelRequest, SetModelResponse
)
from app.models.model_manager import model_manager
from app.services.trainer import train_model_async
from app.utils.logger import logger

router = APIRouter()
