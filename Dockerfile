FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim

ARG SMARTPOLICE_BUILD_COMMIT=unknown
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SMARTPOLICE_BUILD_COMMIT=${SMARTPOLICE_BUILD_COMMIT} \
    PYTHONPATH=/app/backend \
    SMARTPOLICE_DB_PATH=/app/backend/data/smartpolice.db \
    SMARTPOLICE_DATA_ROOT=/app/backend/data \
    SMARTPOLICE_OFFLINE_FIRST=true \
    SMARTPOLICE_ENABLE_CLIP=0 \
    SMARTPOLICE_ENABLE_LOCAL_VLM=0 \
    SMARTPOLICE_REQUIRE_LOCAL_VISION=0 \
    SMARTPOLICE_ENABLE_CLOUD_REVIEW=0 \
    SMARTPOLICE_ENABLE_CLOUD_REPORT=0 \
    ENABLE_DASHSCOPE=0

WORKDIR /app
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8000

CMD python backend/scripts/seed_realistic_demo_cases.py && python backend/scripts/seed_demo_models.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
