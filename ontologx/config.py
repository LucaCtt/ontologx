"""Configuration module for setting up variables used in the log graph building.

The variables are set using environment variables, with default values provided.
Environment variables can be set in the shell before running the script, or in a `.env` file
in the root directory of the project.
"""

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from rdflib import Graph

load_dotenv()

DEFAULT_LLM_MODELS = {
    "ollama": "llama3.2:3b",
    "huggingface": "meta-llama/Llama-3.2-3B-Instruct",
    "vllm": "meta-llama/Llama-3.2-3B-Instruct",
    "bedrock": "meta.llama3-2-3b-instruct-v1:0",
}

DEFAULT_EMBEDDINGS_MODELS = {
    "ollama": "snowflake-arctic-embed:110m",
    "huggingface": "Alibaba-NLP/gte-multilingual-base",
    "infinity": "Alibaba-NLP/gte-multilingual-base",
}

DEFAULT_TESTS_MODELS = {
    "ollama": "llama3.2:3b",
    "huggingface": "meta-llama/Llama-3.2-3B-Instruct",
    "vllm": "meta-llama/Llama-3.2-3B-Instruct",
    "bedrock": "meta.llama3-2-3b-instruct-v1:0",
}


def _get_uri_from_ttl(ttl_path: str) -> str:
    """Get the URI from the TTL file.

    Args:
        ttl_path (str): The path to the TTL file.

    Returns:
        str: The URI of the TTL file.

    """
    graph = Graph()
    graph.parse(ttl_path)
    uri = dict(graph.namespace_manager.namespaces()).get("")
    if uri is None:
        msg = f"Could not find URI in TTL file {ttl_path}"
        raise ValueError(msg)
    return uri.toPython()


class Config:
    """Configuration class for setting up variables used in the log graph building."""

    parser_type = os.getenv("PARSER_TYPE", "main")
    """The type of parser to use. Supported values are 'main' and 'baseline'"""

    examples_retrieval = bool(int(os.getenv("EXAMPLES_RETRIEVAL", "1")))
    """If True, the parser will retrieve examples from the graph store."""

    generated_graphs_retrieval = bool(int(os.getenv("GENERATED_GRAPHS_RETRIEVAL", "1")))
    """If True, only the labelled examples will be used in the RAG."""

    correction_steps = int(os.getenv("CORRECTION_STEPS", "3"))
    """The number of correction steps to take. Must be greater or equal to 0."""

    experiment_name: str = os.getenv("EXPERIMENT_NAME", str(uuid.uuid4()))
    """
    The name of the current experiment. Experiments are used to group runs together,
    the criterion for grouping is up to the user.
    """

    n_runs = int(os.getenv("N_RUNS", "10"))
    """The number of runs to execute in the experiment."""

    ontology_path = os.getenv("ONTOLOGY_PATH", "resources/ontologies/logs.ttl")
    """The path to the ontology file."""

    examples_path = os.getenv("EXAMPLES_PATH", "resources/data/train.ttl")
    """ The path to the examples log graphs file. Used to retrieve the examples."""

    tests_path = os.getenv("TESTS_PATH", "resources/data/test.ttl")
    """The input path to the logs to parse."""

    shacl_path = os.getenv("CONSTRAINTS_PATH", "resources/ontologies/logs_shacl.ttl")
    """The path to the SHACL constraints file for the ontology."""

    prompt_path = os.getenv("PROMPT_PATH", "resources/prompts/build_graph.system.md")
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

    llm_backend = os.getenv("LLM_BACKEND", "ollama")
    """
    The backend to use for the llm.
    Must be one of "ollama", "huggingface", "vllm", or "bedrock".
    """

    llm_backend_url = os.getenv("LLM_BACKEND_URL", "http://localhost:11434")
    """The URL of the llm backend. Used only if the backend is "ollama" or "vllm"."""

    tests_backend = os.getenv("TESTS_BACKEND", "bedrock")
    """The LLM backend to use for the tests. Must be one of the supported LLM backends."""

    tests_backend_url = os.getenv("TESTS_BACKEND_URL", "http://localhost:11434")
    """The URL of the tests backend. Used only if the backend is "ollama" or "vllm"."""

    embeddings_model = os.getenv(
        "EMBEDDINGS_MODEL",
        DEFAULT_EMBEDDINGS_MODELS[embeddings_backend],
    )
    """
    The model used to embed logs. Must be a valid model for the backend used,
    e.g. a model from the HuggingFace model hub if using the HuggingFace backend.
    """

    llm_model = os.getenv(
        "LLM_MODEL",
        DEFAULT_LLM_MODELS[llm_backend],
    )
    """
    The model used to parse logs. Must be a valid model for the backend used,
    e.g. a model from the HuggingFace model hub if using the HuggingFace backend.
    """
    tests_model = os.getenv(
        "TESTS_MODEL",
        DEFAULT_LLM_MODELS[tests_backend],
    )
    """
    The model used to evaluate the tests. Must be a valid model for the backend used,
    e.g. a model from the HuggingFace model hub if using the HuggingFace backend.
    """

    parser_temperature = float(os.getenv("PARSER_TEMPERATURE", "0.7"))
    """The temperature of the LLM used to parse logs. Must be between 0 and 1."""

    events_index_name = os.getenv("EVENTS_INDEX_NAME", "eventMessageIndex")
    """The name of the vector index to use for the events in the graph store."""

    n10s_constraint_name = os.getenv("N10S_CONSTRAINT_NAME", "n10s_unique_uri")
    """The name of the neosemantics constraint for unique URIs."""

    n10s_trigger_name = os.getenv("N10S_TRIGGER_NAME", "shacl_validate")
    """The name of the neosemantics trigger for validating the graph."""

    prompt_build_graph = Path(prompt_path).read_text()
    """The prompt used to build the graph."""

    tests_uri = _get_uri_from_ttl(tests_path)
    """The URI of the tests."""

    ontology_uri = _get_uri_from_ttl(ontology_path)
    """The URI of the ontology."""

    examples_uri = _get_uri_from_ttl(examples_path)
    """The URI of the examples."""

    run_name = str(uuid.uuid4())
    """ The name of the run. A run is a single execution of the ontologx pipeline."""

    run_uri = "https://cyberseclab.unibs.it/ontologx/log/run/" + run_name
    """The URI of the run node."""

    out_uri = "https://cyberseclab.unibs.it/ontologx/log/out/" + run_name
    """The URI of the output nodes."""

    def __init__(self):
        if self.parser_temperature < 0 or self.parser_temperature > 1:
            msg = "Parser temperature must be between 0 and 1"
            raise ValueError(msg)

        if self.correction_steps < 0:
            msg = "Self reflection steps must be greater than 0"
            raise ValueError(msg)

        if self.llm_backend not in DEFAULT_LLM_MODELS:
            msg = f"LLM backend must be one of {DEFAULT_LLM_MODELS.keys()}, but got '{self.llm_backend}'"
            raise ValueError(msg)

        if self.embeddings_backend not in DEFAULT_EMBEDDINGS_MODELS:
            msg = f"Embeddings backend must be one of {DEFAULT_EMBEDDINGS_MODELS.keys()}, \
                but got '{self.embeddings_backend}'"
            raise ValueError(msg)

    def new_run(self) -> None:
        """Create a new run by generating a new UUID and updating the run name."""
        self.run_name = str(uuid.uuid4())
        self.run_uri = "https://cyberseclab.unibs.it/ontologx/log/run/" + self.run_name
        self.out_uri = "https://cyberseclab.unibs.it/ontologx/log/out/" + self.run_name

    def hyperparameters(self) -> dict[str, str | int | float]:
        """Return the hyperparameters used in the experiment.

        Returns:
            dict[str, str]: The hyperparameters used in the experiment.

        """
        return {
            "parser_type": self.parser_type,
            "embeddings_model": self.embeddings_model,
            "parser_model": self.llm_model,
            "parser_temperature": self.parser_temperature,
            "correction_steps": self.correction_steps,
        }
