"""Configuration module for setting up variables used in the log graph building.

The variables are set using environment variables, with default values provided.
Environment variables can be set in the shell before running the script, or in a `.env` file
in the root directory of the project.
"""

import hashlib
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    "groq": "qwen-qwq-32b",
}

DEFAULT_EMBEDDINGS_MODELS = {
    "ollama": "snowflake-arctic-embed:110m",
    "huggingface": "Snowflake/snowflake-arctic-embed-m",
    "google-ai": "gemini-embedding-exp-03-07",
}


class Config:
    """Configuration class for setting up variables used in the log graph building.

    Having a class for this is useful for easily exporting the configuration as a dictionary.
    """

    # The date and time of the experiment.
    experiment_date_time = datetime.now(UTC)

    # Used to distinguish between the data in different experiments.
    experiment_id = os.getenv("EXPERIMENT_ID", str(uuid.uuid4()))

    # The path to the ontology file.
    ontology_path = os.getenv("ONTOLOGY_PATH", "resources/ontologies/logs.ttl")

    # The path to the SHACL constraints file for the ontology.
    shacl_path = os.getenv("CONSTRAINTS_PATH", "resources/ontologies/logs_shacl.ttl")

    # The path to the examples log graphs file.
    examples_path = os.getenv("EXAMPLES_PATH", "resources/data/train.ttl")

    # The input path to the logs to parse.
    tests_path = os.getenv("TEST_LOG_PATH", "resources/data/test.ttl")

    # The prompt used to build the graph
    prompt_build_graph = os.getenv("PROMPT_BUILD_GRAPH", Path("resources/prompts/build_graph.system.md").read_text())

    # Neo4j config
    neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    # The backend to use for the LLM.
    # Must be one of "ollama", "huggingface", or "google-ai".
    backend = os.getenv("BACKEND", "ollama")

    # The HuggingFace hub api token to use for downloading models,
    # generated from https://huggingface.co/docs/hub/security-tokens.
    # Only useful with the HuggingFace backend and when using private models.
    # For public models, this can be left unset.
    # This variable is not used anywhere in the project,
    # it's just to remind that it can be set in the environment.
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", None)

    # The Google AI api key, can be generated from https://ai.google.dev/gemini-api/docs/api-key.
    # Required when using the Google AI backend.
    # This variable is not used anywhere in the project,
    # it's just to remind that it must be set in the environment if using the Google AI backend.
    google_ai_api_key = os.getenv("GOOGLE_API_KEY", None)

    # The model used to embed logs.
    # Must be a model from the HuggingFace model hub if using the HuggingFace backend,
    # or a model from the Ollama model hub if using the Ollama backend.
    embeddings_model = os.getenv(
        "EMBEDDINGS_MODEL",
        DEFAULT_EMBEDDINGS_MODELS[backend],
    )

    # The model used to parse logs.
    # Must be a model from the HuggingFace model hub if using the HuggingFace backend,
    # or a model from the Ollama model hub if using the Ollama backend.
    parser_model = os.getenv(
        "PARSER_MODEL",
        DEFAULT_LLM_MODELS[backend],
    )

    # The temperature of the LLM used to parse logs.
    # Must be between (strictly) 0 and 1.
    parser_temperature = float(os.getenv("PARSER_TEMPERATURE", "0.5"))

    # The number of self-reflection steps to take.
    # Must be greater or equal than 0.
    self_reflection_steps = int(os.getenv("SELF_REFLECTION_STEPS", "3"))

    def __init__(self):
        if self.parser_temperature < 0 or self.parser_temperature > 1:
            msg = "parser_temperature must be between 0 and 1"
            raise ValueError(msg)

        if self.self_reflection_steps < 0:
            msg = "self_reflection_steps must be greater than 0"
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

    def dump(self) -> dict[str, Any]:
        """Dump the configuration as a dictionary.

        This method also includes the ontology and examples hash.
        Authentication params are not included in the dump for security reasons.
        """
        excluded_prefixes = ["_", "neo4j", "huggingface_api_token"]

        dump = {
            key: value
            for key, value in self.__class__.__dict__.items()
            if not any(key.startswith(prefix) for prefix in excluded_prefixes) and not callable(value)
        }
        dump["ontology_hash"] = self.ontology_hash()
        dump["examples_hash"] = self.examples_hash()

        return dump
