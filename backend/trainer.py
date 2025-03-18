import multiprocessing
from app.utils.logger import logger

TIMEOUT_SECONDS = 10

def train_model(model, X_train, y_train, params):
    logger.info(f"Training model {model.id} ({model.name}) with params: {params}")
    model.fit(X_train, y_train, params=params)
    logger.info(f"Model {model.id} ({model.name}) trained successfully!")

# оборачиваем дефолтный train
def train_model_async(model, X_train, y_train, params):
    process = multiprocessing.Process(target=train_model, args=(model, X_train, y_train, params))
    process.start()
    process.join(timeout=TIMEOUT_SECONDS)

    if process.is_alive():
        logger.warning("Training timeout reached. Terminating process...")
        process.terminate()
        process.join()
        logger.info("Process terminated due to timeout.")
