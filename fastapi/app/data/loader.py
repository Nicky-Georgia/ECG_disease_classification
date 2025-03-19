import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from app.utils.logger import logger

warnings.filterwarnings('ignore')

def load_heartbeat_data(test_size=0.2):
    logger.info("Downloading dataset from KaggleHub...")
    path = kagglehub.dataset_download("shayanfazeli/heartbeat")

    mitbih_train = pd.read_csv(path + '/mitbih_train.csv')
    mitbih_test = pd.read_csv(path + '/mitbih_test.csv')

    logger.info("Splitting dataset into train/validation/test...")
    data_train, data_val = train_test_split(np.array(mitbih_train), test_size=test_size, random_state=42)
    
    X_train, y_train = data_train[:, :-1], data_train[:, -1].astype(int)
    X_val, y_val = data_val[:, :-1], data_val[:, -1].astype(int)
    X_test, y_test = np.array(mitbih_test)[:, :-1], np.array(mitbih_test)[:, -1].astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test