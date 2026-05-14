FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --shell /usr/sbin/nologin appuser

COPY --chown=appuser:appuser configs ./configs
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser README.md ./README.md
RUN mkdir -p /app/artifacts && chown -R appuser:appuser /app/artifacts

USER appuser

EXPOSE 9000

CMD ["python", "-m", "uvicorn", "fno_service.api.main:app", "--host", "0.0.0.0", "--port", "9000"]
