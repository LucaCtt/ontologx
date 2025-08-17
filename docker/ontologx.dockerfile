FROM ghcr.io/astral-sh/uv:0.8.11-python3.13-trixie-slim

ENV UV_LINK_MODE=copy

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --group vllm --group aws --no-dev

COPY ontologx ./ontologx
COPY resources ./resources

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENTRYPOINT ["uv", "run", "--no-dev", "olx", "run"]

