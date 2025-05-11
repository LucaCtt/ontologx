from typing import cast

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_neo4j.graphs.graph_document import GraphDocument

from ontologx.parser.models import BaseEventGraph, build_dynamic_model
from ontologx.parser.parser import Parser
from ontologx.parser.tools import fetch_ip_address_info
from ontologx.store import Store


class ToolsParser(Parser):
    """Parser that uses tools to enrich the context of the event and structure output."""

    def __init__(self, llm: BaseChatModel, store: Store, prompt_build_graph: str) -> None:
        super().__init__(llm, store, prompt_build_graph)

        try:
            llm.with_structured_output(BaseEventGraph)
        except NotImplementedError as e:
            msg = "The parser model must support structured output."
            raise ValueError(msg) from e

        gen_graph_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt_build_graph),
                ("human", "Event: '{event}'\nContext: '{context}'"),
            ],
        )

        # Add context enrichment tools.
        # Note: not all models support tools + structured output
        structured_model = llm.bind_tools([fetch_ip_address_info])

        # Add the graph structure to the structured output.
        # Also include raw output to retrieve eventual errors.
        structured_model = structured_model.with_structured_output(  # type: ignore[attr-defined]
            build_dynamic_model(store.ontology()),
            include_raw=True,
        )
        self.chain = gen_graph_prompt | structured_model

    def parse(self, event: str, context: dict) -> GraphDocument | None:
        """Parse the given event and construct a knowledge graph, using tools.

        Args:
            event: The log event to parse.
            context: The context of the event.

        Returns:
            A report containing the stats of the parsing process.

        """
        out = self.chain.invoke({"event": event, "context": context})

        raw_schema = cast(dict, out)

        # Error handling for when the output is not parsed correctly
        if not raw_schema.get("parsed"):
            return None

        output_graph: GraphDocument = raw_schema["parsed"].graph()
        output_graph.source = Document(page_content=event, metadata=context)

        return output_graph
