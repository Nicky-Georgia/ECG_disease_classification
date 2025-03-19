FROM python:3.10-slim

WORKDIR /fastapi/

COPY requirements.txt /fastapi

RUN pip install -r requirements.txt --no-cache-dir

COPY . /fastapi

EXPOSE 8000

# Указываем команду для запуска приложения
CMD ["uvicorn", "app.main:app","--reload", "--host", "0.0.0.0", "--port", "8000"]