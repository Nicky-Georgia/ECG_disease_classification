from pydantic import BaseModel, Field
from typing import List, Any, Dict

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