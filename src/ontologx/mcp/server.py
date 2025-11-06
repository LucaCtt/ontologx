"""OntoLogX MCP Server."""

import asyncio
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from ontologx.backend.llm import LLMFactory
from ontologx.config import Config
from ontologx.parser import MainParser
from ontologx.store import GraphDocument

mcp = FastMCP("OntoLogX")

config = Config()

llm = LLMFactory.create(
    backend=config.parser_backend,
    model=config.parser_model,
    temperature=config.parser_temperature,
    url=config.parser_backend_url,
)
prompt = Path(config.parser_prompt_path).read_text()


@mcp.tool()
async def build_knowledge_graph(
    message: str,
    ontology: GraphDocument,
    context: dict | None = None,
    examples: list[GraphDocument] | None = None,
) -> GraphDocument | None:
    """Build an ontology-compliant knowledge graph from the given message, context, and examples."""
    parser = MainParser(llm, prompt, ontology, 3)

    # Run the parser in a separate thread to avoid blocking the event loop
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, parser.parse, message, context, examples)


if __name__ == "__main__":
    mcp.run()
