"""Backend implementations for generating embeddings and parsing text."""

from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


class Backend(ABC):
    """Abstract base class for backend implementations.

    This class defines the interface for backend services that provide
    embeddings and parser models.
    """

    @abstractmethod
    def embeddings(self, model: str) -> Embeddings:
        """Retrieve embeddings from the specified model.

        Args:
            model (str): The name or identifier of the model from which to get embeddings.

        Returns:
            Embeddings: The embeddings retrieved from the specified model.

        """

    @abstractmethod
    def llm(self, model: str, temperature: float) -> BaseChatModel:
        """Retrieve a parser model based on the specified parameters.

        Args:
            model (str): The name or identifier of the model to retrieve.
            temperature (float): The temperature parameter for the model, which controls the randomness of the output.

        Returns:
            BaseChatModel: A parser model based on the specified parameters.

        """


class BackendFactory:
    """Factory class for creating backend instances based on the specified backend type."""

    @staticmethod
    def create(backend_type: str) -> Backend:
        """Create a backend instance based on the specified backend type.

        Args:
            backend_type (str): The type of backend to create.
            Supported types include "huggingface", "ollama", and "google-ai".

        Returns:
            Backend: An instance of the specified backend type.

        Raises:
            ValueError: If the specified backend type is not supported.

        """
        if backend_type == "huggingface":
            return HuggingFaceBackend()

        if backend_type == "ollama":
            return OllamaBackend()

        if backend_type == "google-ai":
            return GoogleAIBackend()

        msg = f"Unsupported backend type: {backend_type}"
        raise ValueError(msg)


class HuggingFaceBackend(Backend):
    """A backend implementation that uses Hugging Face models for generating embeddings and parsing text."""

    def embeddings(self, model: str) -> Embeddings:
        try:
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(model_name=model, model_kwargs={"trust_remote_code": True})
        except ModuleNotFoundError as e:
            msg = "Please install langchain-huggingface to use HuggingFaceBackend"
            raise ImportError(msg) from e

    def llm(self, model: str, temperature: float) -> BaseChatModel:
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

            parser_pipeline = HuggingFacePipeline.from_model_id(
                model_id=model,
                task="text-generation",
                device_map="auto",
                pipeline_kwargs={
                    "temperature": temperature,
                },
            )
            return ChatHuggingFace(llm=parser_pipeline)
        except ModuleNotFoundError as e:
            msg = "Please install langchain-huggingface to use HuggingFaceBackend"
            raise ImportError(msg) from e


class OllamaBackend(Backend):
    """A backend implementation that uses Ollama models for generating embeddings and parsing text."""

    def embeddings(self, model: str) -> Embeddings:
        try:
            from langchain_ollama.embeddings import OllamaEmbeddings  # type: ignore[import]

            return OllamaEmbeddings(model=model)
        except ModuleNotFoundError as e:
            msg = "Please install langchain-ollama to use OllamaBackend"
            raise ImportError(msg) from e

    def llm(self, model: str, temperature: float) -> BaseChatModel:
        try:
            from langchain_ollama.chat_models import ChatOllama  # type: ignore[import]

            return ChatOllama(model=model, temperature=temperature, num_ctx=1024 * 12)
        except ModuleNotFoundError as e:
            msg = "Please install langchain-ollama to use OllamaBackend"
            raise ImportError(msg) from e


class GoogleAIBackend(Backend):
    """A backend implementation that uses Google AI models for generating embeddings and parsing text."""

    def embeddings(self, model: str) -> Embeddings:
        try:
            from langchain_googleai.embeddings import GoogleAIEmbeddings  # type: ignore[import]

            return GoogleAIEmbeddings(model=model)
        except ModuleNotFoundError as e:
            msg = "Please install langchain-googleai to use GoogleAIBackend"
            raise ImportError(msg) from e

    def llm(self, model: str, temperature: float) -> BaseChatModel:
        try:
            from langchain_googleai import ChatGoogleAI  # type: ignore[import]

            return ChatGoogleAI(model=model, temperature=temperature)
        except ModuleNotFoundError as e:
            msg = "Please install langchain-googleai to use GoogleAIBackend"
            raise ImportError(msg) from e
