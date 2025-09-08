# Pull the uv image to avoid an additional download step
FROM ghcr.io/astral-sh/uv:0.8.11-python3.13-trixie-slim

WORKDIR /app

# Copy the lock files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install required dependencies.
# The `--no-install-project` flag is used to avoid installing the project itself,
# which is handled in a later step to avoid re-downloading dependencies
# when the source code changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --group aws --group openai --no-dev

COPY resources resources
COPY LICENSE LICENSE
COPY src src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --group bedrock --group openai --no-dev

ENTRYPOINT ["uv", "run", "--no-dev", "olx"]
