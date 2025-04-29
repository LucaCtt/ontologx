from langchain_core.language_models import BaseChatModel

from ontologx.parser.baseline_parser import BaselineParser
from ontologx.parser.main_parser import MainParser
from ontologx.parser.parser import Parser
from ontologx.parser.tools_parser import ToolsParser
from ontologx.store import Store


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
