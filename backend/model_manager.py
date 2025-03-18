import os
import joblib
import gdown
from app.utils.logger import logger

MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Google Drive file ID и имя
MODEL_FILES = [
    {
        "id": "1XxXxXxXxXxXxXx",  # <-- Сюда вставь настоящий file_id!!!!!!
        "filename": "default_model.pkl"
    }
]

class ModelManager:
    def __init__(self):
        self.models = {}  # {model_id: модель}
        self.active_model_id = None

    def download_model(self, file_id, filename):
        logger.info(f"Downloading {filename} from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        output = os.path.join(MODELS_DIR, filename)

        # Скачиваем только если файла нет
        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)
            logger.info(f"Downloaded model {filename} to {output}")
        else:
            logger.info(f"Model {filename} already exists locally.")

        return output

    def load_models(self):
        logger.info("Loading models...")

        for model_info in MODEL_FILES:
            file_id = model_info["id"]
            filename = model_info["filename"]

            model_path = self.download_model(file_id, filename)

            try:
                model = joblib.load(model_path)
                model_id = os.path.splitext(filename)[0]

                self.models[model_id] = model
                logger.info(f"Model {model_id} loaded successfully.")

            except Exception as e:
                logger.error(f"Failed to load model {filename}: {e}")

        # Устанавливаем активную модель
        if self.models:
            self.active_model_id = next(iter(self.models))
            logger.info(f"Active model set to: {self.active_model_id}")
        else:
            logger.warning("No models loaded. Active model is None.")

    def get_models_list(self):
        return [
            {
                "id": model_id,
                "is_active": model_id == self.active_model_id
            }
            for model_id in self.models.keys()
        ]

    def get_active_model(self):
        if self.active_model_id:
            return self.models.get(self.active_model_id)
        return None

    def set_active_model(self, model_id):
        if model_id in self.models:
            self.active_model_id = model_id
            logger.info(f"Active model switched to: {model_id}")
            return True
        logger.warning(f"Model {model_id} not found.")
        return False

    def add_model(self, model_id, model_object):
        self.models[model_id] = model_object
        logger.info(f"Added new model {model_id} to manager.")

        # Если не было активной модели — ставим новую
        if not self.active_model_id:
            self.active_model_id = model_id
            logger.info(f"Active model set to: {self.active_model_id}")

    def save_model(self, model_id, filename=None):
        if model_id not in self.models:
            logger.warning(f"Model {model_id} not found, cannot save.")
            return False

        filename = filename or f"{model_id}.pkl"
        model_path = os.path.join(MODELS_DIR, filename)

        try:
            joblib.dump(self.models[model_id], model_path)
            logger.info(f"Model {model_id} saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            return False


model_manager = ModelManager()
