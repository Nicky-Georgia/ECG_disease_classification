FROM python:3.10-slim

WORKDIR /streamlit/

COPY requirements.txt /streamlit

RUN pip install -r requirements.txt --no-cache-dir

COPY . /streamlit

EXPOSE 8501

# Указываем команду для запуска приложения
CMD ["streamlit","run","streamlit_app.py"]