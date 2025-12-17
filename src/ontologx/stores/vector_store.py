"""Module for vector store implementation using Milvus."""

from typing import Any
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

        client = weaviate.connect_to_local(host=host, port=port)
        self.__store = WeaviateVectorStore(
            client=client,
            embedding=embeddings,
            index_name="EventStore",
            text_key="text",
        )

    def add_event(self, event: str, metadata: dict[str, Any]) -> None:
        """Add a single event to the vector store.

        Args:
            event (str): The event string to add.
            metadata (dict[str, Any]): Metadata associated with the event.

        """
        self.add_events([(event, metadata)])

    def add_events(self, events: list[tuple[str, dict[str, Any]]]) -> None:
        """Add events to the vector store.

        Args:
            events (list[tuple[str, dict[str, Any]]]): List of tuples containing event strings and their metadata.

        """
        texts = [f"event: {event}\nmetadata: {metadata}" for event, metadata in events]
        documents = [Document(page_content=text) for text in texts]
        ids = [str(uuid4()) for _ in events]

        self.__store.add_documents(documents, ids=ids)

    def search(self, event: str, metadata: dict[str, Any]) -> list[str]:
        """Search for similar events in the vector store.

        Args:
            event (str): The event string to search for.
            metadata (dict[str, Any]): Metadata associated with the event.

        Returns:
            list[Document]: List of similar event documents.

        """
        text = f"event: {event}\nmetadata: {metadata}"
        results = self.__store.max_marginal_relevance_search(text, alpha=0.5)
        return [doc.page_content for doc in results]
