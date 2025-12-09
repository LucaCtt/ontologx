"""Configuration module for setting experiment parameters.

The variables are set using environment variables, with default values provided.
Environment variables can be set in the shell before running the script,
or in a `.env` file in the root directory of the project.
"""

from pydantic import Field
from pydantic_settings import BaseSettings

_DEFAULT_LLMS = {
    "openai": "gpt-oss-20b",
    "ollama": "llama3.2:3b",
    "bedrock": "meta.llama3-2-3b-instruct-v1:0",
}

_DEFAULT_EMBEDDINGS = {
    "ollama": "milkey/gte",
    "infinity": "Alibaba-NLP/gte-multilingual-base",
}


class Settings(BaseSettings):
    """Configuration class for setting up variables used in the log graph building."""

    correction_steps: int = Field(default=3, ge=0)
    """The number of correction steps to take."""

    ontology_path: str = "resources/ontologies/ontology.rdf"
    """The path to the ontology file."""

    examples_path: str = "resources/data/examples.ttl"
    """ The path to the starter examples graphs file."""

    logs_path: str = "resources/data/ait/logs.ttl"
    """The input path to the logs to parse."""

    embeddings_backend: str = Field(default="infinity", pattern="^(ollama|infinity)$")
    """The backend to use for the embeddings."""

    embeddings_backend_url: str | None = None
    """The URL of the embeddings backend."""

    embeddings_name: str = _DEFAULT_EMBEDDINGS[embeddings_backend]
    """
    The model used to embed logs.
    """

    llm_backend: str = Field(default="bedrock", pattern="^(ollama|openai|bedrock)$")
    """
    The backend to use for the llm.
    Must be one of "ollama", "openai", or "bedrock". Default is "bedrock".
    """

    llm_backend_url: str | None = None
    """The URL of the llm backend."""

    llm_name: str = _DEFAULT_LLMS[llm_backend]
    """The name of the llm to use."""

    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    """The temperature of the llm. Must be between 0 and 1."""

    db_url: str = "http://localhost:7200"
    """The URL of the graph database. """
