import streamlit as st
import requests
import pandas as pd
import json

# Конфигурация
API_URL = "http://localhost:8501"  

def get_models():
    response = requests.get(f"{API_URL}/models")
    if response.status_code == 200:
        return response.json()["models"]
    else:
        st.error("Ошибка при получении списка моделей")
        return []

def set_active_model(model_id):
    response = requests.post(f"{API_URL}/set", json={"model_id": model_id})
    if response.status_code == 200:
        st.success(response.json()["detail"])
    else:
        st.error("Ошибка при установке активной модели")

# Функция для обучения модели
def train_model(hyperparameters):
    response = requests.post(f"{API_URL}/fit", json=hyperparameters)
    if response.status_code == 200:
        st.success("Модель успешно обучена")
    else:
        st.error("Ошибка при обучении модели")

# Функция для предсказания
def predict(data):
    response = requests.post(f"{API_URL}/predict", json={"data": data})
    if response.status_code == 200:
        return response.json()["predictions"]
    else:
        st.error("Ошибка при предсказании")
        return []
    
def fine_tune(data,params={}):
    response = requests.post(f'{API_URL}/finetune',data,params)
    if response.status_code == 200:
        st.success("Модель успешно обучена")
    else:
        st.error("Ошибка при предсказании")
        return []
# Основной интерфейс Streamlit
st.title("Streamlit ML Service")

# Загрузка моделей
models = get_models()
st.write("Доступные модели:", models)

# Обучение модели
st.header("Обучение модели")
hyperparameters = st.text_area("Введите гиперпараметры в формате JSON")
if st.button("Обучить модель"):
    try:
        hyperparams_dict = json.loads(hyperparameters)
        train_model(hyperparams_dict)
    except json.JSONDecodeError:
        st.error("Некорректный формат JSON")

# Предсказание
st.header("Предсказание")
input_data = st.text_area("Введите данные для предсказания в формате JSON")
if st.button("Предсказать"):
    try:
        data_dict = json.loads(input_data)
        prediction = predict(data_dict)
        st.write("Предсказание:", prediction)
    except json.JSONDecodeError:
        st.error("Некорректный формат JSON")