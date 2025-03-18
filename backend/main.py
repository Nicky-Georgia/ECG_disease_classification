from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api import endpoints
from app.models.model_manager import ModelManager
from app.utils.logger import setup_logger

logger = setup_logger()

model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up and loading models...")
    model_manager.load_models_on_startup()
    yield
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan, title="ML Model Deployment API")

app.include_router(endpoints.router)

@app.on_event("startup")
async def startup_event():
    logger.info("Server starting up...")

    # Загружаем модели
    model_manager.load_models()
