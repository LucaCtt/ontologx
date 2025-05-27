"""Backend implementations for generating embeddings and parsing text."""

import os

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


def hf_embeddings(model: str) -> Embeddings:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings  # type: ignore[attr-defined]

    return HuggingFaceEmbeddings(model_name=model, model_kwargs={"trust_remote_code": True})


def infinity_embeddings(model: str, url: str) -> Embeddings:
    from langchain_community.embeddings import InfinityEmbeddings  # type: ignore[attr-defined]

    return InfinityEmbeddings(model=model, infinity_api_url=url)


def ollama_embeddings(model: str, url: str) -> Embeddings:
    from langchain_ollama.embeddings import OllamaEmbeddings  # type: ignore[import]

    return OllamaEmbeddings(model=model, base_url=url)


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

    return ChatOllama(model=model, base_url=url, temperature=temperature)


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
        DurationSeconds=60 * 60 * 3,  # 3 hours
    )

    # Extract the temporary credentials
    credentials = response["Credentials"]

    return ChatBedrockConverse(
        model=model,
        temperature=temperature,
        region_name="us-east-1",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )


class LangchainTestsBackend(DeepEvalBaseLLM):
    def __init__(self, model_name: str, backend: str, url: str = "") -> None:
        self.model_name = model_name
        self.backend = backend
        self.url = url
        self.model = self.load_model()

    def load_model(self) -> BaseChatModel:  # type: ignore[override]
        return BackendFactory.create_llm(backend=self.backend, model=self.model_name, url=self.url, temperature=0.4)

    def generate(self, prompt: str) -> str:
        out = self.model.invoke(prompt)
        return str(out.content)

    async def a_generate(self, prompt: str) -> str:
        out = await self.model.ainvoke(prompt)
        return str(out.content)

    def get_model_name(self) -> str:
        return f"{self.model_name} ({self.backend})"


class BackendFactory:
    @classmethod
    def create_llm(
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
            case "huggingface":
                return hf_llm(model, temperature)

            case "ollama":
                return ollama_llm(model, temperature, url)

            case "vllm":
                return vllm_llm(model, temperature, url)

            case "bedrock":
                return bedrock_llm(model, temperature)

            case _:
                msg = f"Unsupported backend type: {backend}"
                raise ValueError(msg)

    @classmethod
    def create_embeddings(
        cls,
        backend: str,
        model: str,
        url: str = "",
    ) -> Embeddings:
        """Create an embeddings instance based on the specified backend type.

        Args:
            backend (str): The backend to use for creating embeddings.
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

            case "infinity":
                return infinity_embeddings(model, url)

            case "ollama":
                return ollama_embeddings(model, url)

            case _:
                msg = f"Unsupported backend type: {backend}"
                raise ValueError(msg)

    @classmethod
    def create_tests(
        cls,
        backend: str,
        model: str,
        url: str = "",
    ) -> DeepEvalBaseLLM:
        """Create a test instance based on the specified backend type.

        Args:
            backend (str): The backend to use for creating tests.
            model (str): The name or identifier of the model to use.
            url (str): The URL for the language model.

        Returns:
            DeepEvalBaseLLM: An instance of the specified backend type.

        Raises:
            ValueError: If the specified backend type is not supported.

        """
        return LangchainTestsBackend(model_name=model, backend=backend, url=url)
