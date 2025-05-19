import os
import joblib
import gdown
from app.utils.logger import logger

MODELS_DIR = "app/models"

class ModelManager:
    '''class of model managing'''
    def __init__(self):
        self.models = {}
        self.active_model_id = None

    def load_models(self):
        """
        Загружает модели из локальной папки MODELS_DIR при старте приложения.
        """
        logger.info(f"Loading models from {MODELS_DIR}...")

        if not os.path.exists(MODELS_DIR):
            logger.warning(f"{MODELS_DIR} does not exist. Creating...")
            os.makedirs(MODELS_DIR)

        for filename in os.listdir(MODELS_DIR):
            if filename.endswith(".joblib"):
                model_id = filename.split(".")[0]
                model_path = os.path.join(MODELS_DIR, filename)
                try:
                    model = joblib.load(model_path)
                    self.models[model_id] = model
                    logger.info(f"Model '{model_id}' loaded from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load model '{model_id}': {str(e)}")

        if self.models:
            self.active_model_id = next(iter(self.models))
            logger.info(f"Active model set to {self.active_model_id}")

    def download_and_load_model_from_gdrive(self, file_id, filename="pretrained_model.joblib"):
        """
        Скачивает модель из Google Drive.
        """
        destination = os.path.join(MODELS_DIR, filename)

        logger.info(f"Downloading model from Google Drive (file_id={file_id}) to {destination}...")

        url = r"https://drive.google.com/uc?id=1G5VIxNP8l80l7o_KK6T_QrSvFB8vJa1b"

        try:
            gdown.download(url, destination, quiet=False)
            logger.info(f"Model downloaded successfully to {destination}")

            model_id = filename.split(".")[0]
            model = joblib.load(destination)
            self.models[model_id] = model
            logger.info(f"Model '{model_id}' loaded and added to model manager")

            if not self.active_model_id:
                self.active_model_id = model_id
                logger.info(f"Active model set to {self.active_model_id}")

        except Exception as e:
            logger.error(f"Failed to download or load model: {str(e)}")

    def get_models_info(self):
        """
        Возвращает информацию о всех доступных моделях.
        """
        return [
            {"id": model_id, "name": str(type(model)), 
            "is_active": model_id == self.active_model_id}
            for model_id, model in self.models.items()
        ]

    def set_active_model(self, model_id):
        """
        Устанавливает активную модель по её id.
        """
        if model_id in self.models:
            self.active_model_id = model_id
            logger.info(f"Active model changed to {model_id}")
            return True
        logger.warning(f"Model {model_id} not found!")
        return False

    def get_active_model(self):
        """
        Возвращает активную модель.
        """
        if self.active_model_id:
            return self.models[self.active_model_id]
        logger.warning("No active model selected!")
        return None

    def add_model(self, model_id, model_object):
        """
        Добавление модели в менеджер с проверкой корректности.
        """
        if hasattr(model_object, 'fit'):
            self.models[model_id] = model_object
            logger.info(f"Model {model_id} added to model manager.")
        else:
            logger.error(f"Cannot add model '{model_id}' to manager. Model is not valid.")
            raise ValueError(f"Model '{model_id}' is not valid, it cannot be added to the manager.")

model_manager = ModelManager()
