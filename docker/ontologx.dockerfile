FROM python:3.13-slim

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        curl \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry==2.1.0

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN poetry install --with vllm, ollama && rm -rf ${POETRY_CACHE_DIR}

COPY ontologx ./ontologx

RUN poetry run python -m olx clear

ENTRYPOINT ["poetry", "run", "python", "-m", "olx", "run"]

