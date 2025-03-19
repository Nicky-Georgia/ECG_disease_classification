import streamlit as st
import requests
import pandas as pd
import json

# Конфигурация
API_URL = "http://localhost:8000"  

# Функция для загрузки модели
def load_model():
    response = requests.get(f"{API_URL}/models")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Ошибка при загрузке моделей")
        return []

# Функция для обучения модели
def train_model(hyperparameters):
    response = requests.post(f"{API_URL}/fit", json=hyperparameters)
    if response.status_code == 200:
        st.success("Модель успешно обучена")
    else:
        st.error("Ошибка при обучении модели")

# Функция для предсказания
def predict(data):
    response = requests.post(f"{API_URL}/predict", json=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Ошибка при предсказании")
        return None

# Функция для загрузки датасета
def upload_dataset(file):
    files = {"file": file.getvalue()}
    response = requests.post(f"{API_URL}/upload", files=files)
    if response.status_code == 200:
        st.success("Датасет успешно загружен")
    else:
        st.error("Ошибка при загрузке датасета")

# Основной интерфейс Streamlit
st.title("Streamlit ML Service")

# Загрузка моделей
models = load_model()
st.write("Доступные модели:", models)

# Загрузка датасета
st.header("Загрузка датасета")
uploaded_file = st.file_uploader("Выберите файл с датасетом")
if uploaded_file is not None:
    upload_dataset(uploaded_file)

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