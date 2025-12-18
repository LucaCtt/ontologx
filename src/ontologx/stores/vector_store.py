"""Module for vector store implementation using Milvus."""

from urllib.parse import urlparse
from uuid import uuid4

import weaviate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_weaviate import WeaviateVectorStore


class VectorStore:
    """Vector store implementation using Weaviate."""

    def __init__(self, embeddings: Embeddings, url: str) -> None:
        parsed_url = urlparse(url)
        host = parsed_url.hostname
        port = parsed_url.port
        if host is None or port is None:
            msg = f"Invalid URL: {url}"
            raise ValueError(msg)

        self.__client = weaviate.connect_to_local(host=host, port=port)
        self.__store = WeaviateVectorStore(
            client=self.__client,
            embedding=embeddings,
            index_name="EventStore",
            text_key="text",
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

    def search(self, event: str) -> list[str]:
        """Search for similar events in the vector store.

        Args:
            event (str): The event string to search for.

        Returns:
            list[Document]: List of similar event documents.

        """
        results = self.__store.max_marginal_relevance_search(event, alpha=0.5)
        return [doc.page_content for doc in results]

    def close(self) -> None:
        """Close the vector store connection."""
        self.__client.close()
