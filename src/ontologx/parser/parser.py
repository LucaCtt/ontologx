"""Parser module for parsing messages and constructing knowledge graphs."""

import logging
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from ontologx.store import GraphDocument

logger = logging.getLogger("rich")


class Parser(ABC):
    """Parser class for parsing messages and constructing knowledge graphs using a LLM."""

    def __init__(self, llm: BaseChatModel, prompt_build_graph: str, ontology: GraphDocument) -> None:
        self.llm = llm
        self.prompt_build_graph = prompt_build_graph
        self.ontology = ontology

    @abstractmethod
    def parse(
        self,
        message: str,
        context: dict | None = None,
        examples: list[GraphDocument] | None = None,
    ) -> GraphDocument | None:
        """Parse the given message and construct a knowledge graph.

        Args:
            message: The message to parse.
            context: The context of the event.
            examples: A list of example GraphDocuments to guide the parsing.

        Returns:
            A GraphDocument representing the constructed knowledge graph, or None if parsing failed.

        """
