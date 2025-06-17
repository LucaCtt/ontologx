# OntoLogX

This repository contains the code for OntoLogX, developed by researchers at the University of Brescia under the NEACD project.

## Getting Started

### Installation
To install OntoLogX, you can use Poetry. First, make sure you have Poetry installed. Then, run the following command:

```bash
poetry install --with dev, vllm, aws
```
This will create a virtual environment and install all the main project dependencies, including the development dependencies and the optional dependencies for the VLLM and AWS backends. See [Backends](#backends) for more information on the supported backends.

### Running OntoLogX

The easiest way to run OntoLogX is to use one of the Docker compose files, which only require a working docker installation. Two docker compose configurations can be found in the `docker` directory: one for the vLLM backend and one for the AWS Bedrock backend. Both the configurations use [Infinity](https://github.com/michaelfeil/infinity) for the embeddings backend.

Create an appropriate `.env` file with the minimum values specified in the [Backends](#backends) section for the backend you want to use. For example, to use the vLLM backend:
```env
echo "LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
EMBEDDINGS_MODEL=Alibaba-NLP/gte-multilingual-base
TESTS_MODEL=meta-llama/Meta-Llama-3.3-70B-Instruct
" > .env
```

Then, you can run the following command to start the compose:
```bash
docker compose -f docker/compose.vllm.yaml --env-file .env up --build --abort-on-container-exit
```

## Configuration
The project configuration is specified through environment variables. You can set them in a `.env` file in the root directory of the project, where you can also find a `.env.default` with an example configuration. The following sections describe the available environment variables and their default values.

Experiment configuration. Should almost always be set manually:
- `PARSER_TYPE`: The type of parser to use for the input files. Supported values are `baseline` for a prompt-only parser without any enhancements, or `main` fo the OntoLogX parser which supports all the features of the project. Default is `main`.
- `EXAMPLES_RETRIEVAL`: Whether to use the examples retrieval feature. Relevant only for the `main` parser. Supported values are 0 (for no retrieval) or 1 (for retrieval). Default is 1.
- `GENERATED_GRAPHS_RETRIEVAL`: Whether to use the generated graphs retrieval feature. Relevant only for the `main` parser and if `EXAMPLES_RETRIEVAL` is set to 1. Supported values are 0 (for no retrieval) or 1 (for retrieval). Default is 1.
- `CORRECTION_STEPS`: The number of correction steps to perform during the generation phase. Relevant only for the `main` parser. Default is 3.
- `EXPERIMENT_NAME`: The name of the experiment to run. This will be used to create an Experiment node in the Neo4j database. Default is a random UUID.
- `N_RUNS`: The number of runs to perform for the experiment. Default is 10.
- `EMBEDDINGS_MODEL`: The model to use for the embeddings. This should be a valid model name for the embeddings backend you are using. Default is `gte-multilingual-base`.
- `LLM_MODEL`: The model to use for the LLM calls. This should be a valid model name for the LLM backend you are using. Default is `Llama 3.2 3B`.
- `TESTS_MODEL`: The model to use for the tests. This should be a valid model name for the tests backend you are using. Default is `Llama 3.2 3B`. A larger model than the default is recommended.
- `PARSER_TEMPERATURE`: The temperature to use for the parser. This is a float value between 0 and 1, where 0 means no randomness and 1 means maximum randomness. Default is 0.7.

Backends configuration. No need to set these manually if you are using the provided Docker Compose configurations, as they will be set automatically based on the backend you choose. If you are running OntoLogX locally, you should set these variables according to your backend configuration:
- `NEO4J_URI`: The URI of the Neo4j database (default: `bolt://localhost:7687`).
- `NEO4J_USERNAME`: The username for the Neo4j database (default: `neo4j`).
- `NEO4J_PASSWORD`: The password for the Neo4j database (default: `password`).
- `EMBEDDINGS_BACKEND`: The backend to use for the embeddings. Supported values are `ollama`, `huggingface`, or `infinity`. Default is `ollama`.
- `EMBEDDINGS_BACKEND_URL`: The URL of the embeddings backend, relevant only for `ollama` and `infinity` backends. Default is `http://localhost:11434`.
- `LLM_BACKEND`: The backend to use for the LLM calls. Supported values are `vllm`, `aws`, `ollama`, or `huggingface`. Default is `vllm`.
- `LLM_BACKEND_URL`: The URL of the LLM backend, relevant only for `ollama` and `vllm` backends. Default is `http://localhost:11434`.
- `TESTS_BACKEND`: The backend to use for the tests. Supported values are `vllm`, `aws`, `ollama`, or `huggingface`. Default is `aws`.
- `TESTS_BACKEND_URL`: The URL of the tests backend, relevant only for `ollama` and `vllm` backends. Default is `http://localhost:11434`.

TTL paths and prompts configuration. You should not need to change them unless you want to use custom files:
- `ONTOLOGY_PATH`: The path to the ontology file to use for the logs. This should be a valid TTL file. Default is `resources/ontologies/logs.ttl`.
- `EXAMPLES_PATH`: The path to the examples file to use for the logs. This should be a valid TTL file. Default is `resources/logs/train.ttl`.
- `TESTS_PATH`: The path to the tests file to use for the logs. This should be a valid TTL file. Default is `resources/logs/test.ttl`.
- `SHACL_PATH`: The path to the SHACL file to use for the logs. This should be a valid TTL file. Default is `resources/ontologies/logs-shacl.ttl`.
- PROMPT_PATH: The path to the prompt file to use for the generation phase. This should be a valid TTL file. Default is `resources/prompts/main.md` if `PARSER_TYPE` is set to `main`, or `resources/prompts/baseline.md` if `PARSER_TYPE` is set to `baseline`.
- `EVENTS_INDEX_NAME`: Name of the Neo4j vector index. Default is `eventMessageIndex`.
- `N10S_CONSTRAINT_NAME`: Name of the Neo4j constraint for the N10S plugin. Default is `n10s_unique_uri`.
- `N10S_TRIGGER_NAME`: Name of the Neo4j trigger for the N10S plugin SHACL validation. Default is `shacl_validate`.

## Backends

The LLM calls in OntoLogX are abstracted through a common interface, allowing you to switch between different LLM backends without changing the code. It is possible to use different backends for the generation and test phases, by using appropriate environment variables (see [Configuration](#configuration)). Currently, the following LLM backends are implemented:
- [vLLM](https://vllm.ai/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Ollama](https://ollama.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)

Because of the requirement of a high-quality LLM for tests, it is recommended to use AWS Bedrock with an appropriate model, such as Llama 3.3, Claude 3.5 Sonnet, or better. Note that Bedrock is the default test backend when using the provided Docker Compose configurations.

For tests you can use any of the LLM backends, but it is recommended to use a large, state-of-the-art model.

For the embeddings, OntoLogX supports the following backends:
- [Infinity](https://github.com/michaelfeil/infinity)
- [Ollama](https://ollama.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)

It is recommended to use Infinity for the embeddings, as it provides a fast and efficient way to compute embeddings. The default embeddings model is `gte-multilingual-base`, which is a good choice for general-purpose embeddings.

## Development

If you intend to work on the codebase, first make sure to install the development dependencies by running:

```bash
poetry install --with dev
```

To ensure code quality and consistency, we use [Ruff](https://beta.ruff.rs/docs/) for linting. The `pyproject.toml` file contains the configuration for Ruff, which is recommende for linting and formatting the code. Make suure to install the appropriate extension for your editor, such as the [Ruff extension for VSCode](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff). To run the linter manually, you can use the following command:

```bash
poetry run ruff check .
```

The repo also includes pre-commit hooks for running Ruff, and to check the Poetry config. To install them, run:
```bash
pre-commit install
```

### Useful Queries

This section reports useful queries to run in the Neo4j database to inspect the data generated by OntoLogX.

- Get run result nodes:
```
MATCH (n) WHERE n:EvaluationMeasure or n:ModelEvaluation or n:Run RETURN n
```
This command is particulary useful to inspect the number of completed runs, after launching an experiment.

- Get mean evaluation results with standard deviation:
```
MATCH (m:EvaluationMeasure)<-[:specifiedBy]-(e:ModelEvaluation) RETURN m.hasName AS measure, avg(e.hasValue) AS mean, stDev(e.hasValue) AS SD ORDER BY measure
```

## License

MIT. See [LICENSE](/LICENSE) for details.

## Author

Luca Cotti <luca.cotti@unibs.it>

