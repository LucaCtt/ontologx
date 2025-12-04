"""Configuration module for setting experiment parameters.

The variables are set using environment variables, with default values provided.
Environment variables can be set in the shell before running the script,
or in a `.env` file in the root directory of the project.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_LLMS = {
    "openai": "gpt-oss-20b",
    "ollama": "llama3.2:3b",
    "bedrock": "meta.llama3-2-3b-instruct-v1:0",
}

_DEFAULT_EMBEDDINGS = {
    "ollama": "milkey/gte",
    "infinity": "Alibaba-NLP/gte-multilingual-base",
}


@dataclass(frozen=True)
class Config:
    """Configuration class for setting up variables used in the log graph building.

    This class is immutable and uses environment variables to set its attributes on instantiation.
    Reassigning attributes after instantiation will raise an error.
    """

    experiment_name: str = os.getenv("EXPERIMENT_NAME", "default")
    """
    The name of the current experiment. Experiments are used to group runs together,
    the criterion for grouping is up to the user.
    """

    n_runs = int(os.getenv("N_RUNS", "1"))
    """The number of runs to execute in the experiment."""

    correction_steps = int(os.getenv("CORRECTION_STEPS", "3"))
    """The number of correction steps to take. Must be greater or equal to 0."""

    ontology_path = os.getenv("ONTOLOGY_PATH", "resources/ontologies/uco_1_5_owl.rdf")
    """The path to the ontology file."""

    examples_path = os.getenv("EXAMPLES_PATH", "resources/data/examples.ttl")
    """ The path to the starter examples graphs file."""

    logs_path = os.getenv("TESTS_PATH", "resources/data/ait/logs.ttl")
    """The input path to the logs to parse."""

    builder_prompt_path = os.getenv("BUILDER_PROMPT_PATH", "resources/prompts/builder.system.md")
    """The path of the prompt used to build the graph."""

    tactics_prompt_path = os.getenv(
        "TACTICS_PROMPT_PATH",
        "resources/prompts/tactics.system.md",
    )
    """The path of the prompt used to predict tactics."""

    embeddings_backend = os.getenv("EMBEDDINGS_BACKEND", "infinity")
    """
    The backend to use for the embeddings.
    Must be "ollama" or "infinity".
    Default is "infinity".
    """

    embeddings_backend_url = os.getenv("EMBEDDINGS_BACKEND_URL", "http://localhost:11434")
    """The URL of the embeddings backend. Used only if the backend is "ollama" or "infinity"."""

    embeddings_name = os.getenv(
        "EMBEDDINGS_NAME",
        _DEFAULT_EMBEDDINGS[embeddings_backend],
    )
    """
    The model used to embed logs.
    """

    llm_backend = os.getenv("LLM_BACKEND", "bedrock")
    """
    The backend to use for the llm.
    Must be one of "ollama", "openai", or "bedrock". Default is "bedrock".
    """

    llm_backend_url = os.getenv("LLM_BACKEND_URL", "http://localhost:11434")
    """The URL of the llm backend. Used only if the backend is "ollama" or "vllm"."""

    llm_name = os.getenv(
        "LLM_NAME",
        _DEFAULT_LLMS[llm_backend],
    )
    """The name of the llm to use."""

    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    """The temperature of the llm. Must be between 0 and 1."""

    db_url = os.getenv("DB_URL", "http://localhost:7200")
    """The URL of the graph database. """
