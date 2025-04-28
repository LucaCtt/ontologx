"""Parser module for parsing log events and constructing knowledge graphs."""

import logging
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_neo4j.graphs.graph_document import GraphDocument

from ontologx.parser.baseline_parser import BaselineParser
from ontologx.parser.main_parser import MainParser
from ontologx.parser.tools_parser import ToolsParser
from ontologx.store import Store

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


class ParserFactory:
    """Factory class for creating Parser instances."""

    @staticmethod
    def create(
        parser_type: str,
        llm: BaseChatModel,
        store: Store,
        prompt_build_graph: str,
        correction_steps: int = 0,
    ) -> Parser:
        """Create a Parser instance.

        Args:
            parser_type: The type of parser to create. Can be "baseline", "tools", or "main".
            llm: The language model to use for parsing.
            store: The store to use for storing the parsed graphs.
            prompt_build_graph: The prompt to use for building the graph.
            correction_steps: The number of correction steps to perform.
            **kwargs: Additional arguments to pass to the parser constructor.

        Returns:
            A Parser instance.

        """
        match parser_type:
            case "baseline":
                return BaselineParser(llm, store, prompt_build_graph)
            case "tools":
                return ToolsParser(llm, store, prompt_build_graph)
            case "main":
                return MainParser(llm, store, prompt_build_graph, correction_steps)
            case _:
                msg = f"Unknown parser type: {parser_type}"
                raise ValueError(msg)
