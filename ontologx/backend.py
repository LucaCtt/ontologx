"""Backend implementations for generating embeddings and parsing text."""

import os
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


def hf_embeddings(model: str) -> Embeddings:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # type: ignore[attr-defined]

    return HuggingFaceEmbeddings(model_name=model, model_kwargs={"trust_remote_code": True})


def ollama_embeddings(model: str) -> Embeddings:
    from langchain_ollama.embeddings import OllamaEmbeddings  # type: ignore[import]

    return OllamaEmbeddings(model=model)


def google_ai_embeddings(model: str) -> Embeddings:
    from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings  # type: ignore[import]

    return GoogleGenerativeAIEmbeddings(model=model)


class EmbeddingsFactory:
    @staticmethod
    def create(backend: str, model: str, **kwargs: dict[str, Any]) -> Embeddings:
        """Create an embeddings instance based on the specified backend type.

        Args:
            backend(str): The backend to use for creating embeddings.
            model (str): The name or identifier of the model to use.
            **kwargs: Additional keyword arguments for the embeddings instance.

        Returns:
            Embeddings: An instance of the specified backend type.

        Raises:
            ValueError: If the specified backend type is not supported.

        """
        match backend:
            case "huggingface":
                return hf_embeddings(model, **kwargs)

            case "ollama":
                return ollama_embeddings(model, **kwargs)

            case "google-ai":
                return google_ai_embeddings(model, **kwargs)

            case _:
                msg = f"Unsupported backend type: {backend}"
                raise ValueError(msg)


def hf_llm(model: str, temperature: float) -> BaseChatModel:
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline  # type: ignore[attr-defined]

    parser_pipeline = HuggingFacePipeline.from_model_id(
        model_id=model,
        task="text-generation",
        device_map="auto",
        pipeline_kwargs={
            "temperature": temperature,
        },
    )
    return ChatHuggingFace(llm=parser_pipeline)


def ollama_llm(model: str, temperature: float) -> BaseChatModel:
    from langchain_ollama.chat_models import ChatOllama  # type: ignore[import]

    return ChatOllama(model=model, temperature=temperature, num_ctx=1024 * 12)


def google_ai_llm(model: str, temperature: float) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import]

    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def bedrock_llm(model: str, temperature: float) -> BaseChatModel:
    import boto3
    from langchain_aws import ChatBedrockConverse  # type: ignore[import]

    sts_client = boto3.client("sts")

    # Assume the role
    response = sts_client.assume_role(
        RoleArn="arn:aws:sts::816558913136:role/Bedrock",
        RoleSessionName="langchain-bedrock-session",
        DurationSeconds=60 * 60 * 3,  # 3 hours
    )

    # Extract the temporary credentials
    credentials = response["Credentials"]
    os.environ["AWS_ACCESS_KEY_ID"] = credentials["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["SecretAccessKey"]
    os.environ["AWS_SESSION_TOKEN"] = credentials["SessionToken"]

    return ChatBedrockConverse(
        model=model,
        region_name="us-east-1",
        temperature=temperature,
    )


class LLMFactory:
    """Factory class for creating backend instances based on the specified backend type."""

    @staticmethod
    def create(backend: str, model: str, temperature: float) -> BaseChatModel:
        """Create an LLM instance based on the specified backend type.

        Args:
            backend (str): The backend to use for creating the LLM.
            model (str): The name or identifier of the model to use.
            temperature (float): The temperature setting for the LLM.

        Returns:
            BaseChatModel: An instance of the specified backend type.

        Raises:
            ValueError: If the specified backend type is not supported.

        """
        if backend == "huggingface":
            return hf_llm(model, temperature)

        if backend == "ollama":
            return ollama_llm(model, temperature)

        if backend == "google-ai":
            return google_ai_llm(model, temperature)

        if backend == "bedrock":
            return bedrock_llm(model, temperature)

        msg = f"Unsupported backend type: {backend}"
        raise ValueError(msg)
