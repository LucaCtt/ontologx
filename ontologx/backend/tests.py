"""Factory for creating test instances based on backend type."""

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.language_models import BaseChatModel

from ontologx.backend.llm import LLMFactory


class TestsFactory:
    """Factory class for creating test models for DeepEval."""

    @classmethod
    def create(
        cls,
        backend: str,
        model: str,
        url: str = "",
    ) -> DeepEvalBaseLLM:
        """Create an instance of DeepEvalBaseLLM for testing purposes.

        Args:
            backend (str): The LLM backend to use for the model.
            model (str): The name of the LLM to use.
            url (str): The URL for the backend, if applicable.

        Returns:
            DeepEvalBaseLLM: An instance of LangchainTestsBackend configured with the specified parameters.

        """

        class LangchainTestsBackend(DeepEvalBaseLLM):
            def __init__(self) -> None:
                self.model = self.load_model()

            def load_model(self) -> BaseChatModel:  # type: ignore[override]
                return LLMFactory.create(backend=backend, model=model, url=url, temperature=0.4)

            def generate(self, prompt: str) -> str:
                out = self.model.invoke(prompt)
                return str(out.content)

            async def a_generate(self, prompt: str) -> str:
                out = await self.model.ainvoke(prompt)
                return str(out.content)

            def get_model_name(self) -> str:
                return f"{model} ({backend})"

        return LangchainTestsBackend()
