"""Parser module for parsing log events and constructing knowledge graphs."""

import logging
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from ontologx.store import GraphDocument, Store

logger = logging.getLogger("rich")


class Parser(ABC):
    """Parser class for parsing log events and constructing knowledge graphs using a LLM."""

    def __init__(self, llm: BaseChatModel, store: Store, prompt_build_graph: str) -> None:
        self.llm = llm
        self.store = store
        self.prompt_build_graph = prompt_build_graph

    @abstractmethod
    def parse(self, event: str, context: dict) -> GraphDocument | None:
        """Parse the given event and construct a knowledge graph.

        Args:
            event: The log event to parse.
            context: The context of the event.

        Returns:
            A report containing the stats of the parsing process.

        """
        # Retrieve examples once for all the self-reflection steps
