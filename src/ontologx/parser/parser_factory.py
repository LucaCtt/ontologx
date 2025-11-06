"""Parser factory module."""

from langchain_core.language_models import BaseChatModel

from ontologx.parser.baseline_parser import BaselineParser
from ontologx.parser.main_parser import MainParser
from ontologx.parser.parser import Parser
from ontologx.store import GraphDocument


class ParserFactory:
    """Factory class for creating Parser instances."""

    @staticmethod
    def create(
        parser_type: str,
        llm: BaseChatModel,
        prompt_build_graph: str,
        ontology: GraphDocument,
        correction_steps: int = 0,
    ) -> Parser:
        """Create a Parser instance.

        Args:
            parser_type: The type of parser to create. Can be "baseline" or "main".
            llm: The language model to use for parsing.
            prompt_build_graph: The prompt to use for building the graph.
            ontology: The ontology graph document.
            correction_steps: The number of correction steps to perform.

        Returns:
            A Parser instance.

        """
        match parser_type:
            case "baseline":
                return BaselineParser(llm, prompt_build_graph, ontology)
            case "main":
                return MainParser(llm, prompt_build_graph, ontology, correction_steps)
            case _:
                msg = f"Unknown parser type: {parser_type}"
                raise ValueError(msg)
