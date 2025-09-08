"""Factory for creating embeddings instances based on backend type."""

from langchain_core.embeddings import Embeddings


def infinity_embeddings(model: str, url: str) -> Embeddings:
    """Create an Infinity embeddings instance."""
    from langchain_community.embeddings import InfinityEmbeddings  # type: ignore[attr-defined]

    return InfinityEmbeddings(model=model, infinity_api_url=url)


def ollama_embeddings(model: str, url: str) -> Embeddings:
    """Create an Ollama embeddings instance."""
    from langchain_ollama.embeddings import OllamaEmbeddings  # type: ignore[import]

    return OllamaEmbeddings(model=model, base_url=url)


class EmbeddingsFactory:
    """Factory class for creating embeddings instances based on backend type."""

    @classmethod
    def create(
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
            case "infinity":
                return infinity_embeddings(model, url)

            case "ollama":
                return ollama_embeddings(model, url)

            case _:
                msg = f"Unsupported backend type: {backend}"
                raise ValueError(msg)
