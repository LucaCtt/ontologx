"""Backend implementations for generating embeddings and parsing text."""

import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


def hf_embeddings(model: str) -> Embeddings:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # type: ignore[attr-defined]

    return HuggingFaceEmbeddings(model_name=model, model_kwargs={"trust_remote_code": True})


def ollama_embeddings(model: str, url: str) -> Embeddings:
    from langchain_ollama.embeddings import OllamaEmbeddings  # type: ignore[import]

    return OllamaEmbeddings(model=model, base_url=url)


def vllm_embeddings(model: str, url: str) -> Embeddings:
    from langchain_openai.embeddings import OpenAIEmbeddings  # type: ignore[import]

    return OpenAIEmbeddings(
        model=model,
        base_url=url,
        model_kwargs={"trust_remote_code": True},
    )


class EmbeddingsFactory:
    @staticmethod
    def create(backend: str, model: str, url: str) -> Embeddings:
        """Create an embeddings instance based on the specified backend type.

        Args:
            backend(str): The backend to use for creating embeddings.
            model (str): The name or identifier of the model to use.
            url (str): The URL for the backend.

        Returns:
            Embeddings: An instance of the specified backend type.

        Raises:
            ValueError: If the specified backend type is not supported.

        """
        match backend:
            case "huggingface":
                return hf_embeddings(model)

            case "ollama":
                return ollama_embeddings(model, url)

            case "vllm":
                return vllm_embeddings(model, url)

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


def ollama_llm(model: str, temperature: float, url: str) -> BaseChatModel:
    from langchain_ollama.chat_models import ChatOllama  # type: ignore[import]

    return ChatOllama(model=model, url=url, temperature=temperature, num_ctx=1024 * 12)


def vllm_llm(model: str, temperature: float, url: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI  # type: ignore[import]

    return ChatOpenAI(
        model=model,
        base_url=url,
        temperature=temperature,
    )


def bedrock_llm(model: str, temperature: float) -> BaseChatModel:
    import boto3  # type: ignore[attr-defined]
    from botocore.config import Config  # type: ignore[attr-defined]
    from langchain_aws import ChatBedrockConverse  # type: ignore[import]

    sts_client = boto3.client("sts", config=Config(read_timeout=300))

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
    def create(backend: str, model: str, temperature: float, url: str) -> BaseChatModel:
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
        if backend == "huggingface":
            return hf_llm(model, temperature)

        if backend == "ollama":
            return ollama_llm(model, temperature, url)

        if backend == "vllm":
            return vllm_llm(model, temperature, url)

        if backend == "bedrock":
            return bedrock_llm(model, temperature)

        msg = f"Unsupported backend type: {backend}"
        raise ValueError(msg)
