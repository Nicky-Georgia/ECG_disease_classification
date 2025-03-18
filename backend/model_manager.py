import uuid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from app.models.base_model import BaseModel
from app.services.dataset_loader import load_heartbeat_data
from app.utils.logger import logger

class ModelManager:
    def __init__(self):
        self.models = {}
        self.active_model_id = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def load_models_on_startup(self):
        logger.info("Loading heartbeat dataset...")
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = load_heartbeat_data()

        logger.info("Initializing models...")
        self._register_model(DecisionTreeClassifier(), "DecisionTree")
        self._register_model(RandomForestClassifier(), "RandomForest")
        self._register_model(GradientBoostingClassifier(), "GradientBoosting")

        logger.info("All models registered.")
        logger.info(f"Active model set to {self.active_model_id}")

    def _register_model(self, model, model_type):
        model_id = str(uuid.uuid4())
        wrapped_model = BaseModel(model_id=model_id, model_type=model_type, model=model)
        self.models[model_id] = wrapped_model
        
        if self.active_model_id is None:
            self.active_model_id = model_id
        
        logger.info(f"Model {model_type} ({model_id}) registered.")

    def list_models(self):
        return [
            {
                "id": model.id,
                "name": model.name,
                "description": f"{model.name} model",
                "is_active": model.id == self.active_model_id
            }
            for model in self.models.values()
        ]

    def set_active_model(self, model_id):
        if model_id in self.models:
            self.active_model_id = model_id
            logger.info(f"Active model switched to {model_id}")
            return True
        logger.warning(f"Model {model_id} not found")
        return False

    def get_active_model(self):
        return self.models.get(self.active_model_id)

    def get_data_for_training(self):
        return self.X_train, self.y_train

    def get_data_for_prediction(self):
        return self.X_test, self.y_test
