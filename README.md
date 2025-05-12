# OntoLogX

This repository contains the code for OntoLogX, developed by researchers at the University of Brescia under the NEACD project.

## Getting Started

### Installation
To install OntoLogX, you can use Poetry. First, make sure you have Poetry installed. Then, run the following command:

```bash
poetry install
```
This will create a virtual environment and install all the dependencies listed in the `pyproject.toml` file.

## Configuration
The project configuration is specified through environment variables. You can set them in a `.env` file in the root directory of the project. The following environment variables are supported:

## Development

## Linting and Formatting
To ensure code quality and consistency, we use [Ruff](https://beta.ruff.rs/docs/) for linting. To run the linter, use the following command:  

```bash
poetry run ruff check
```

If you use [VSCode](https://code.visualstudio.com/), you can install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) to get real-time linting feedback in your editor.

The Ruff configuration is located in the `pyproject.toml` file under the `[tool.ruff]` section.

### Pre-commit hooks

The repo includes pre-commit hooks for running Ruff, and to check the Poetry config. To install them, run:

```bash
pre-commit install
```

### Useful Queries

Here are some useful neo4j queries:

- Get run result nodes:
```
MATCH (n) WHERE n:EvaluationMeasure or n:ModelEvaluation or n:Run RETURN n
```

- Get evaluation results:
```
MATCH (m:EvaluationMeasure)<-[:specifiedBy]-(e:ModelEvaluation) RETURN m.hasName AS measure, avg(e.hasValue) AS averageValue, stDev(e.hasValue) AS stdDeviation ORDER BY measure
```

## License

MIT. See [LICENSE](/LICENSE) for details.

## Author

Luca Cotti <luca.cotti@unibs.it>

