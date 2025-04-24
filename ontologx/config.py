"""Configuration module for setting up variables used in the log graph building.

The variables are set using environment variables, with default values provided.
Environment variables can be set in the shell before running the script, or in a `.env` file
in the root directory of the project.
"""

import hashlib
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _compute_file_hash(file_path: str) -> str:
    """Compute the SHA256 hash of a file."""
    with Path(file_path).open("rb", buffering=0) as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


DEFAULT_LLM_MODELS = {
    "ollama": "qwen2.5-coder:14b",
    "huggingface": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "google-ai": "gemini-2.0-flash",
}

DEFAULT_EMBEDDINGS_MODELS = {
    "ollama": "snowflake-arctic-embed:110m",
    "huggingface": "Snowflake/snowflake-arctic-embed-m",
    "google-ai": "models/text-embedding-004",
}


class Config:
    """Configuration class for setting up variables used in the log graph building."""

    experiment_name: str = os.getenv("EXPERIMENT_NAME", "Default")
    """
    The name of the current experiment. Experiments are used to group runs together,
    the criterion for grouping is up to the user.
    """

    ontology_path = os.getenv("ONTOLOGY_PATH", "resources/ontologies/logs.ttl")
    """The path to the ontology file."""

    ontology_uri = os.getenv("ONTOLOGY_URI", "https://cyberseclab.unibs.it/ontologx/log/dictionary")
    """The URI of the ontology."""

    examples_path = os.getenv("EXAMPLES_PATH", "resources/data/train.ttl")
    """ The path to the examples log graphs file. Used to retrieve the examples."""

    examples_uri = os.getenv("EXAMPLES_URI", "https://cyberseclab.unibs.it/ontologx/log/examples/1.0")
    """The URI of the examples."""

    tests_path = os.getenv("TESTS_PATH", "resources/data/test.lc.ttl")
    """The input path to the logs to parse."""

    tests_uri = os.getenv("TESTS_URI", "https://cyberseclab.unibs.it/ontologx/log/tests/lc/1.0")
    """The URI of the tests."""

    run_name = os.getenv("RUN_NAME", str(uuid.uuid4()))
    """ The name of the run. A run is a single execution of the ontologx pipeline."""

    run_uri = os.getenv("RUN_URI", "https://cyberseclab.unibs.it/ontologx/log/run/" + run_name)
    """The URI of the nodes generated in the run."""

    out_uri = os.getenv("OUT_URI", "https://cyberseclab.unibs.it/ontologx/log/out/" + run_name)

    shacl_path = os.getenv("CONSTRAINTS_PATH", "resources/ontologies/logs_shacl.ttl")
    """The path to the SHACL constraints file for the ontology."""

    prompt_build_graph = os.getenv("PROMPT_BUILD_GRAPH", Path("resources/prompts/build_graph.system.md").read_text())
    """The prompt used to build the graph."""

    neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    """The URL of the Neo4j database. Use bolt+ssc for self-signed certificates."""

    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    """The username to use for the Neo4j database."""

    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    """The password to use for the Neo4j database."""

    backend = os.getenv("BACKEND", "ollama")
    """
    The backend to use for the LLM.
    Must be one of "ollama", "huggingface", or "google-ai".
    """

    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", None)
    """
    The HuggingFace hub api token to use for downloading models,
    generated from https://huggingface.co/docs/hub/security-tokens.
    Only useful with the HuggingFace backend and when using private models.
    For public models, this can be left unset.
    This variable is not used anywhere in the project,
    it's just to remind that it can be set in the environment.
    """

    google_ai_api_key = os.getenv("GOOGLE_API_KEY", None)
    """
    The Google AI api key, can be generated from https://ai.google.dev/gemini-api/docs/api-key.
    Required when using the Google AI backend.
    This variable is not used anywhere in the project,
    it's just to remind that it must be set in the environment if using the Google AI backend.
    """

    embeddings_model = os.getenv(
        "EMBEDDINGS_MODEL",
        DEFAULT_EMBEDDINGS_MODELS[backend],
    )
    """
    The model used to embed logs. Must be a valid model for the backend used,
    e.g. a model from the HuggingFace model hub if using the HuggingFace backend.
    """

    parser_model = os.getenv(
        "PARSER_MODEL",
        DEFAULT_LLM_MODELS[backend],
    )
    """
    The model used to parse logs. Must be a valid model for the backend used,
    e.g. a model from the HuggingFace model hub if using the HuggingFace backend.
    """

    parser_temperature = float(os.getenv("PARSER_TEMPERATURE", "0.5"))
    """The temperature of the LLM used to parse logs. Must be between 0 and 1."""

    self_reflection_steps = int(os.getenv("SELF_REFLECTION_STEPS", "3"))
    """The number of self-reflection steps to take. Must be greater or equal to 0."""

    events_index_name = os.getenv("EVENTS_INDEX_NAME", "eventMessageIndex")
    """The name of the vector index to use for the events in the graph store."""

    n10s_constraint_name = os.getenv("N10S_CONSTRAINT_NAME", "n10s_unique_uri")
    """The name of the neosemantics constraint for unique URIs."""

    n10s_trigger_name = os.getenv("N10S_TRIGGER_NAME", "shacl-validate")
    """The name of the neosemantics trigger for validating the graph."""

    def __init__(self):
        if self.parser_temperature < 0 or self.parser_temperature > 1:
            msg = "Parser temperature must be between 0 and 1"
            raise ValueError(msg)

        if self.self_reflection_steps < 0:
            msg = "Self reflection steps must be greater than 0"
            raise ValueError(msg)

        if self.backend not in ["ollama", "huggingface", "google-ai"]:
            msg = f"backend must be one of 'ollama', 'huggingface', or 'google-ai', but got '{self.backend}'"
            raise ValueError(msg)

        if self.backend == "google-ai" and self.google_ai_api_key is None:
            msg = "GOOGLE_API_KEY must be set in the environment when using the Google AI backend"
            raise ValueError(msg)

    def ontology_hash(self) -> str:
        return _compute_file_hash(self.ontology_path)

    def examples_hash(self) -> str:
        return _compute_file_hash(self.examples_path)

    def tests_hash(self) -> str:
        return _compute_file_hash(self.tests_path)

    def hyperparameters(self) -> dict[str, str]:
        """Return the hyperparameters used in the experiment.

        Returns:
            dict[str, str]: The hyperparameters used in the experiment.

        """
        return {
            "parser_temperature": str(self.parser_temperature),
            "self_reflection_steps": str(self.self_reflection_steps),
        }
