FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GRIDSIGHT_DATA_DIR=/data GRIDSIGHT_REPORTS_DIR=/data/reports
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY dashboard ./dashboard
RUN pip install --no-cache-dir -e ".[dashboard]"
VOLUME ["/data"]
EXPOSE 8000 8501
CMD ["gridsight", "serve", "--host", "0.0.0.0", "--port", "8000"]
