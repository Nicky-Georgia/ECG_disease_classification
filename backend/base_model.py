from sklearn.metrics import f1_score
from app.utils.logger import logger

class BaseModel:
    def __init__(self, model_id, model_type, model):
        self.id = model_id
        self.name = model_type
        self.model = model
        self.is_trained = False

    def fit(self, X_train, y_train, params=None):
        logger.info(f"Training {self.name} model {self.id}...")

        if params:
            logger.info(f"Updating model params: {params}")
            self.model.set_params(**params)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info(f"{self.name} model {self.id} trained.")

    def predict(self, X):
        if not self.is_trained:
            logger.warning(f"Model {self.id} has not been trained yet!")
            raise ValueError("Model not trained yet.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        score = f1_score(y_test, preds, average='micro')
        logger.info(f"Evaluation of {self.name} model {self.id}: F1 Score = {score}")
        return score
