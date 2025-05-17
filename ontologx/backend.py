"""Backend implementations for generating embeddings and parsing text."""

import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel


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
    os.environ["AWS_ACCESS_KEY_ID"] = credentials["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["SecretAccessKey"]
    os.environ["AWS_SESSION_TOKEN"] = credentials["SessionToken"]

    return ChatBedrockConverse(
        model=model,
        region_name="us-east-1",
        temperature=temperature,
    )


def bedrock_tests(model: str) -> DeepEvalBaseLLM:
    class AWSBedrock(DeepEvalBaseLLM):
        def __init__(
            self,
            model_name: str,
        ):
            self.model_name = self.load_model(model_name)

        def load_model(self, model: str) -> BaseChatModel:  # type: ignore[override]
            return bedrock_llm(model, 0.4)

        def generate(self, prompt: str, schema: BaseModel) -> BaseModel:  # type: ignore[override]
            structured_llm = self.model.with_structured_output(schema)  # type: ignore[attr-defined]
            return structured_llm.invoke(prompt)  # type: ignore[return-value]

        async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:  # type: ignore[override]
            structured_llm = self.model.with_structured_output(schema)  # type: ignore[attr-defined]
            return await structured_llm.ainvoke(prompt)  # type: ignore[return-value]

        def get_model_name(self) -> str:
            return f"{self.model_name} (AWS Bedrock)"

    return AWSBedrock(model)


class BackendFactory:
    def create_llm(
        self,
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

    def create_embeddings(
        self,
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
