from typing import List, Any, Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    '''class for predictions'''
    data: List[float] = Field(..., description="Input features for prediction")

class PredictResponse(BaseModel):
    '''class for response'''
    prediction: List[Any] = Field(..., description="Predicted class or values")

class FitRequest(BaseModel):
    '''class for fitting'''
    params: Dict[str, Any] = Field(..., description="Hyperparameters for training")

class FitResponse(BaseModel):
    '''class for fitting response'''
    message: str

class UploadResponse(BaseModel):
    '''class for upload response'''
    message: str

class SetModelRequest(BaseModel):
    '''class for model request'''
    model_id: str

class SetModelResponse(BaseModel):
    '''class for model response'''
    message: str

class ModelsListResponse(BaseModel):
    '''class for model list response'''
    models: List[Dict[str, Any]]

class ArchitectureEnum(str, Enum):
    '''class for architecture'''
    random_forest = "RandomForest"
    gradient_boosting = "GradientBoosting"

class CreateModelRequest(BaseModel):
    '''class for model request'''
    architecture: ArchitectureEnum = Field(..., description="Type of model architecture")
    model_id: str = Field(..., description="Unique ID for the model")
    params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Hyperparameters for the model"
        )
