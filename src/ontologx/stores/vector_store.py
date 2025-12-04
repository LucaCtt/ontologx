"""Module for vector store implementation using Milvus."""

from uuid import uuid4

from langchain.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus


class VectorStore:
    """Vector store implementation using Milvus."""

    def __init__(self, embeddings: Embeddings, uri: str = "./ontologx_vector.db") -> None:
        self.__store = Milvus(
            embedding_function=embeddings,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            connection_args={"uri": uri},
            index_params={"index_type": "FLAT", "metric_type": "cosine"},
        )

    def add_event(self, event: str) -> None:
        """Add a single event to the vector store.

        Args:
            event (str): The event string to add.

        """
        self.add_events([event])

    def add_events(self, events: list[str]) -> None:
        """Add events to the vector store.

        Args:
            events (list[str]): List of event strings to add.

        """
        documents = [Document(page_content=event) for event in events]
        ids = [str(uuid4()) for _ in events]

        self.__store.add_documents(documents, ids=ids)

    def search(self, event: str, k: int = 5) -> list[str]:
        """Search for similar events in the vector store.

        Args:
            event (str): The event string to search for.
            k (int, optional): Number of similar events to retrieve. Defaults to 5.

        Returns:
            list[Document]: List of similar event documents.

        """
        results = self.__store.similarity_search(event, k=k, ranker_type="rrf", ranker_params={"k": 60})
        return [doc.page_content for doc in results]
