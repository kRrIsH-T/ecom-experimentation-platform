# Pipeline / API / dashboard image (Linux, so no winutils needed).
FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ECOM_ROOT=/app

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .
COPY configs ./configs
COPY dashboard ./dashboard
COPY reports/templates ./reports/templates

EXPOSE 8000 8501
CMD ["ecom", "--help"]
