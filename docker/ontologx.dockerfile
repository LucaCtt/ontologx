FROM python:3.13-slim

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        curl \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./
COPY ontologx ./ontologx
RUN touch README.md

RUN poetry install --with ollama
