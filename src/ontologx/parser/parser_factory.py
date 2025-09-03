"""Parser factory module."""

from langchain_core.language_models import BaseChatModel

from ontologx.parser.baseline_parser import BaselineParser
from ontologx.parser.main_parser import MainParser
from ontologx.parser.parser import Parser
from ontologx.store import Store


class ParserFactory:
    """Factory class for creating Parser instances."""

    @staticmethod
    def create(
        parser_type: str,
        llm: BaseChatModel,
        store: Store,
        prompt_build_graph: str,
        examples_retrieval: bool,
        correction_steps: int = 0,
    ) -> Parser:
        """Create a Parser instance.

        Args:
            parser_type: The type of parser to create. Can be "baseline" or "main".
            llm: The language model to use for parsing.
            store: The store to use for storing the parsed graphs.
            prompt_build_graph: The prompt to use for building the graph.
            examples_retrieval: Whether to use examples retrieval.
            correction_steps: The number of correction steps to perform.
            **kwargs: Additional arguments to pass to the parser constructor.

        Returns:
            A Parser instance.

        """
        match parser_type:
            case "baseline":
                return BaselineParser(llm, store, prompt_build_graph, examples_retrieval)
            case "main":
                return MainParser(llm, store, prompt_build_graph, correction_steps, examples_retrieval)
            case _:
                msg = f"Unknown parser type: {parser_type}"
                raise ValueError(msg)
