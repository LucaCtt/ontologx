"""Factory for creating LLM instances based on backend type."""

import os

from langchain_core.language_models import BaseChatModel


def ollama_llm(model: str, temperature: float, url: str) -> BaseChatModel:
    """Create an Ollama LLM instance using the specified model and URL."""
    from langchain_ollama.chat_models import ChatOllama  # type: ignore[import]

    return ChatOllama(model=model, base_url=url, temperature=temperature)


def bedrock_llm(model: str, temperature: float) -> BaseChatModel:
    """Create a Bedrock LLM instance using temporary credentials from AWS STS."""
    import boto3  # type: ignore[import]
    from botocore.config import Config  # type: ignore[import]
    from langchain_aws import ChatBedrockConverse  # type: ignore[import]

    sts_client = boto3.client(
        "sts",
        config=Config(read_timeout=300),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    # Assume the role
    response = sts_client.assume_role(
        RoleArn=os.environ["AWS_ROLE_ARN"],
        RoleSessionName="langchain-bedrock-session",
        DurationSeconds=60 * 60 * 10,  # 10 hours
    )

    # Extract the temporary credentials
    credentials = response["Credentials"]

    return ChatBedrockConverse(
        model=model,
        temperature=temperature,
        region_name=os.environ["AWS_REGION"],
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )


def openai_llm(model: str, temperature: float, url: str) -> BaseChatModel:
    """Create an OpenAI LLM instance using the specified model and URL."""
    from langchain_openai import ChatOpenAI  # type: ignore[import]

    return ChatOpenAI(
        model=model,
        base_url=url,
        temperature=temperature,
    )


class LLMFactory:
    """Factory class for creating LLM instances."""

    @classmethod
    def create(
        cls,
        backend: str,
        model: str,
        temperature: float,
        url: str = "",
    ) -> BaseChatModel:
        """Create an LLM instance based on the specified backend type.

        Args:
            backend (str): The backend to use for creating the LLM.
            model (str): The name or identifier of the model to use.
            temperature (float): The temperature setting for the LLM.
            url (str): The URL for the backend.

        Returns:
            BaseChatModel: An instance of the specified backend type.

        Raises:
            ValueError: If the specified backend type is not supported.

        """
        match backend:
            case "ollama":
                return ollama_llm(model, temperature, url)

            case "bedrock":
                return bedrock_llm(model, temperature)

            case "openai":
                return openai_llm(model, temperature, url)

            case _:
                msg = f"Unsupported backend type: {backend}"
                raise ValueError(msg)
