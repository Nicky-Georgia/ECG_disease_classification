from fastapi import FastAPI
from app.routers import endpoints
from app.utils.logger import logger
from app.managers.model_manager import model_manager

app = FastAPI(title="ECG Classification Models Deployment API")

app.include_router(endpoints.router)

@app.on_event("startup")
async def startup_event():
    logger.info("Server starting up...")

    model_manager.download_and_load_model_from_gdrive(
        file_id="X",
        filename="heartbeat_model.joblib"
    )

    model_manager.load_models()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Server shutting down...")