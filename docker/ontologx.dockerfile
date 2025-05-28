FROM python:3.13-slim

RUN pip install poetry==2.1.0

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN touch README.md

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --with vllm,aws --no-root

COPY ontologx ./ontologx
COPY resources ./resources

RUN poetry install

ENTRYPOINT ["poetry", "run", "olx", "run"]

