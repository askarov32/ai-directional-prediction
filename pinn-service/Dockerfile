FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY README.md ./README.md
COPY artifacts ./artifacts

EXPOSE 9000

CMD ["uvicorn", "pinn_service.service_app:app", "--host", "0.0.0.0", "--port", "9000"]
