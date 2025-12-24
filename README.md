# OntoLogX

This repository contains the code for OntoLogX, developed by researchers at the University of Brescia under the NEACD project.

## Getting Started

### Development

If you intend to work on the codebase, first make sure to install the development dependencies by running:

```bash
uv sync
```

To ensure code quality and consistency, we use [Ruff](https://beta.ruff.rs/docs/) for linting. The `pyproject.toml` file contains the configuration for Ruff, which is recommende for linting and formatting the code. Make suure to install the appropriate extension for your editor, such as the [Ruff extension for VSCode](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff). To run the linter manually, you can use the following command:

```bash
uv run ruff check .
```

The repo also includes pre-commit hooks for running Ruff, and to check the `uv` config. To install them, run:

```bash
pre-commit install
```

### Running OntoLogX

The easiest way to run OntoLogX is to use one of the Docker compose files, which only require a working docker installation. A docker compose configurations can be found in the `docker` directory, which uses AWS Bedrock for LLM calls and [Infinity](https://github.com/michaelfeil/infinity) for the embeddings.

You can run the following command to start the compose:

```bash
docker compose -f docker/compose.yaml --env-file .env up --build --abort-on-container-exit
```

## Configuration

The project configuration is specified through environment variables. You can set them in a `.env` file in the root directory of the project and pass them through Docker compose using the `--env-file` option, or you can set them directly in your shell environment if you are running the project locally.

You can find a list of all the configuration variables, along with their descriptions and default values, in the [settings file](/src/ontologx/settings.py) .

## License

MIT. See [LICENSE](/LICENSE) for details.

## Author

Luca Cotti <luca.cotti@unibs.it>
