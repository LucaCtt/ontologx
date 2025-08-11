"""Configuration module for setting up variables used in the log graph building.

The variables are set using environment variables, with default values provided.
Environment variables can be set in the shell before running the script, or in a `.env` file
in the root directory of the project.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_LLM_MODELS = {
    "ollama": "llama3.2:3b",
    "huggingface": "meta-llama/Llama-3.2-3B-Instruct",
    "vllm": "meta-llama/Llama-3.2-3B-Instruct",
    "bedrock": "meta.llama3-2-3b-instruct-v1:0",
}

_DEFAULT_EMBEDDINGS_MODELS = {
    "ollama": "milkey/gte",
    "huggingface": "Alibaba-NLP/gte-multilingual-base",
    "infinity": "Alibaba-NLP/gte-multilingual-base",
}


@dataclass(frozen=True)
class Config:
    """Configuration class for setting up variables used in the log graph building.

    This class is immutable and uses environment variables to set its attributes on instantiation.
    Reassigning attributes after instantiation will raise an error.
    """

    parser_type = os.getenv("PARSER_TYPE", "main")
    """The type of parser to use. Supported values are 'main' and 'baseline'"""

    examples_retrieval = bool(int(os.getenv("EXAMPLES_RETRIEVAL", "1")))
    """If True, the parser will retrieve examples from the graph store."""

    generated_graphs_retrieval = bool(int(os.getenv("GENERATED_GRAPHS_RETRIEVAL", "1")))
    """If True, only the labelled examples will be used in the RAG."""

    correction_steps = int(os.getenv("CORRECTION_STEPS", "3"))
    """The number of correction steps to take. Must be greater or equal to 0."""

    experiment_name: str = os.getenv("EXPERIMENT_NAME", "default")
    """
    The name of the current experiment. Experiments are used to group runs together,
    the criterion for grouping is up to the user.
    """

    n_runs = int(os.getenv("N_RUNS", "10"))
    """The number of runs to execute in the experiment."""

    ontology_path = os.getenv("ONTOLOGY_PATH", "resources/ontologies/logs.ttl")
    """The path to the ontology file."""

    examples_path = os.getenv("EXAMPLES_PATH", "resources/data/ait/train.ttl")
    """ The path to the examples log graphs file. Used to retrieve the examples."""

    tests_path = os.getenv("TESTS_PATH", "resources/data/ait/test.ttl")
    """The input path to the logs to parse."""

    shacl_path = os.getenv("CONSTRAINTS_PATH", "resources/ontologies/logs_shacl.ttl")
    """The path to the SHACL constraints file for the ontology."""

    prompt_path = os.getenv(
        "PROMPT_PATH",
        "resources/prompts/main.system.md" if parser_type == "main" else "resources/prompts/baseline.system.md",
    )
    """The prompt used to build the graph."""

    neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    """The URL of the Neo4j database. Use bolt+ssc for self-signed certificates."""

    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    """The username to use for the Neo4j database."""

    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    """The password to use for the Neo4j database."""

    embeddings_backend = os.getenv("EMBEDDINGS_BACKEND", "ollama")
    """
    The backend to use for the embeddings.
    Must be one of "ollama", "huggingface", or "infinity".
    """

    embeddings_backend_url = os.getenv("EMBEDDINGS_BACKEND_URL", "http://localhost:11434")
    """The URL of the embeddings backend. Used only if the backend is "ollama" or "infinity"."""

    embeddings_model = os.getenv(
        "EMBEDDINGS_MODEL",
        _DEFAULT_EMBEDDINGS_MODELS[embeddings_backend],
    )
    """
    The model used to embed logs. Must be a valid model for the backend used,
    e.g. a model from the HuggingFace model hub if using the HuggingFace backend.
    """

    parser_backend = os.getenv("PARSER_BACKEND", "ollama")
    """
    The backend to use for the parser llm.
    Must be one of "ollama", "huggingface", "vllm", or "bedrock".
    """

    parser_backend_url = os.getenv("PARSER_BACKEND_URL", "http://localhost:11434")
    """The URL of the parser llm backend. Used only if the backend is "ollama" or "vllm"."""

    parser_model = os.getenv(
        "PARSER_MODEL",
        _DEFAULT_LLM_MODELS[parser_backend],
    )
    """
    The model used to parse logs. Must be a valid model for the backend used,
    e.g. a model from the HuggingFace model hub if using the HuggingFace backend.
    """

    parser_temperature = float(os.getenv("PARSER_TEMPERATURE", "0.7"))
    """The temperature of the LLM used to parse logs. Must be between 0 and 1."""

    tests_backend = os.getenv("TESTS_BACKEND", "bedrock")
    """The LLM backend to use for the tests. Must be one of the supported LLM backends."""

    tests_backend_url = os.getenv("TESTS_BACKEND_URL", "http://localhost:11434")
    """The URL of the tests backend. Used only if the backend is "ollama" or "vllm"."""

    tests_model = os.getenv(
        "TESTS_MODEL",
        _DEFAULT_LLM_MODELS[tests_backend],
    )
    """
    The model used to evaluate the tests. Must be a valid model for the backend used,
    e.g. a model from the HuggingFace model hub if using the HuggingFace backend.
    """

    tests_temperature = float(os.getenv("TESTS_TEMPERATURE", "0.4"))
    """The temperature of the LLM used to evaluate the tests. Must be between 0 and 1."""

    def __init__(self):
        if self.parser_temperature < 0 or self.parser_temperature > 1:
            msg = "Parser temperature must be between 0 and 1"
            raise ValueError(msg)

        if self.correction_steps < 0:
            msg = "Self reflection steps must be greater than 0"
            raise ValueError(msg)

        if self.parser_backend not in _DEFAULT_LLM_MODELS:
            msg = f"LLM backend must be one of {_DEFAULT_LLM_MODELS.keys()}, but got '{self.parser_backend}'"
            raise ValueError(msg)

        if self.embeddings_backend not in _DEFAULT_EMBEDDINGS_MODELS:
            msg = f"Embeddings backend must be one of {_DEFAULT_EMBEDDINGS_MODELS.keys()}, \
                but got '{self.embeddings_backend}'"
            raise ValueError(msg)
