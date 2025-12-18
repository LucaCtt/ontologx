"""Configuration module for setting experiment parameters.

The variables are set using environment variables, with default values provided.
Environment variables can be set in the shell before running the script,
or in a `.env` file in the root directory of the project.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration class for setting up variables used in the log graph building."""

    correction_steps: int = Field(default=3, ge=0)
    """The number of correction steps to take."""

    ontology_path: str = "resources/ontologies/logs.ttl"
    """The path to the ontology file."""

    examples_path: str = "resources/data/train"
    """ The path to the starter examples logs file."""

    logs_path: str = "resources/data/validation.csv"
    """The input path to the logs to parse."""

    embeddings_url: str = "http://localhost:8000"
    """The URL of the embeddings backend."""

    embeddings_name: str = "Alibaba-NLP/gte-multilingual-base"
    """The model used to embed logs."""

    llm_url: str | None = None
    """The URL of the llm backend."""

    llm_name: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    """The name of the llm to use."""

    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    """The temperature of the llm. Must be between 0 and 1."""

    graph_database_url: str = "http://localhost:7200"
    """The URL of the graph database. """

    vector_database_url: str = "http://localhost:8080"

    hf_token: SecretStr | None = None

    aws_access_key_id: SecretStr | None = None

    aws_secret_access_key: SecretStr | None = None

    aws_role_arn: SecretStr | None = None

    aws_region: str | None = None
