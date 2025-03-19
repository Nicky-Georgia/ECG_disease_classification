from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional
from enum import Enum

class PredictRequest(BaseModel):
    data: List[float] = Field(..., description="Input features for prediction")

class PredictResponse(BaseModel):
    prediction: List[Any] = Field(..., description="Predicted class or values")

class FitRequest(BaseModel):
    params: Dict[str, Any] = Field(..., description="Hyperparameters for training")

class FitResponse(BaseModel):
    message: str

class UploadResponse(BaseModel):
    message: str

class SetModelRequest(BaseModel):
    model_id: str

class SetModelResponse(BaseModel):
    message: str

class ModelsListResponse(BaseModel):
    models: List[Dict[str, Any]]

class ArchitectureEnum(str, Enum):
    random_forest = "RandomForest"
    gradient_boosting = "GradientBoosting"

class CreateModelRequest(BaseModel):
    architecture: ArchitectureEnum = Field(..., description="Type of model architecture")
    model_id: str = Field(..., description="Unique ID for the model")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Hyperparameters for the model")