FROM python:3.10-slim

WORKDIR /streamlit/

COPY requirements.txt /streamlit

RUN pip install -r requirements.txt --no-cache-dir

COPY . /streamlit

# Указываем команду для запуска приложения
CMD ["streamlit","run","streamlit_app.py"]