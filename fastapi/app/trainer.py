import multiprocessing
from app.utils.logger import logger

TIMEOUT_SECONDS = 10


def train_model(model, X_train, y_train, params):  
    '''model training'''
    logger.info(f"Training model {model} with params: {params}")
    try:
        model.set_params(**params)
        model.fit(X_train, y_train)
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise e

def train_model_async(model, X_train, y_train, params):
    '''async training'''
    process = multiprocessing.Process(target=train_model, args=(model, X_train, y_train, params))
    process.start()
    process.join(timeout=TIMEOUT_SECONDS)

    if process.is_alive():
        logger.warning("Training timeout reached. Terminating process...")
        process.terminate()
        process.join()
        logger.info("Process terminated due to timeout.")
