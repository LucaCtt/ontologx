"""Graph Connector agent for processing and merging graphs."""

import logging
from dataclasses import dataclass, replace

import polars as pl
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from rdflib import Graph

from ontologx.agents.graph_builder.agent import GraphBuilderContext, GraphBuilderInputState, graph_builder_agent
from ontologx.agents.tactics_predictor.agent import (
    TacticsPredictorContext,
    TacticsPredictorInput,
    tactics_predictor_agent,
)
from ontologx.stores import GraphStore, VectorStore

CHUNK_GRAPHS_NODE = "chunk_graphs"
BUILD_GRAPHS_FOR_CHUNK_NODE = "build_graphs_for_chunk"
PREDICT_TTPS_FOR_CHUNK_NODE = "predict_ttps"
SAVE_CHUNK_NODE = "save_chunk"

logger = logging.getLogger("rich")


@dataclass(frozen=True)
class GraphConnectorInput:
    """Input for the Graph Connector agent."""

    # The list of graphs to process
    events: pl.DataFrame


@dataclass(frozen=True)
class _GraphConnectorState(GraphConnectorInput):
    chunks: list[pl.DataFrame]

    current_chunk_index: int = 0


@dataclass(frozen=True)
class GraphConnectorContext:
    """Context for the Graph Connector agent."""

    # The LLM to use for TTP prediction
    llm: BaseChatModel

    # The ontology for the log graphs
    ontology: Graph

    # The vector store to save embeddings
    vector_store: VectorStore

    # The graph store to save the merged graphs
    graph_store: GraphStore

    # Maximum number of graph conversation history to keep in memory
    max_graph_conversation_history: int = 5

    # Maximum number of graph correction steps
    max_graph_correction_steps: int = 3

    max_chunk_size: int = 10


def _chunk_graphs(state: GraphConnectorInput, runtime: Runtime[GraphConnectorContext]) -> _GraphConnectorState:
    # Group by application and device
    chunks = []
    for _, df in state.events.group_by(["application", "device"]):
        if df.height <= runtime.context.max_chunk_size:
            chunks.append(df)
        else:
            chunks.extend(
                [
                    df.slice(i, runtime.context.max_chunk_size)
                    for i in range(0, df.height, runtime.context.max_chunk_size)
                ],
            )

    logger.info("Chunked events into %d chunks", len(chunks))
    logger.info("Chunked events into %d chunks", len(chunks))

    return _GraphConnectorState(
        **state.__dict__,
        chunks=chunks,
    )


def _build_graphs_for_chunk(
    state: _GraphConnectorState,
    runtime: Runtime[GraphConnectorContext],
) -> _GraphConnectorState:
    # Build a graph for the current chunk of events
    current_chunk = state.chunks[state.current_chunk_index]
    logger.info("Building graphs for chunk %d/%d", state.current_chunk_index + 1, len(state.chunks))

    # Reset memory saver for graph builder agent
    graph_builder_agent.checkpointer = MemorySaver()

    graphs = []
    for row in current_chunk.iter_rows(named=True):
        relevant_events = runtime.context.vector_store.search(
            row["log event"],
        )
        logger.info("Found %d relevant events for event '%s'", len(relevant_events), row["log event"])

        relevant_kgs = [runtime.context.graph_store.get_graph(event) for event in relevant_events]
        logger.info("Successfully retrieved %d relevant knowledge graphs", len(relevant_kgs))

        output = graph_builder_agent.invoke(
            GraphBuilderInputState(
                input_event=row["log event"],
                input_examples=relevant_kgs,
            ),
            context=GraphBuilderContext(
                llm=runtime.context.llm,
                ontology=runtime.context.ontology,
                max_conversation_history=runtime.context.max_graph_conversation_history,
                max_correction_steps=runtime.context.max_graph_correction_steps,
            ),
        )
        graph = output["output_graph"]
        graphs.append(graph if graph is not None else Graph())
        logger.info("Built graph for event '%s'", row["log event"])

    # Maybe inefficient
    new_chunks = state.chunks.copy()
    new_chunks[state.current_chunk_index] = new_chunks[state.current_chunk_index].with_columns(
        pl.Series("graph", graphs),
    )
    return replace(state, chunks=new_chunks)


def _predict_ttps_for_chunk(
    state: _GraphConnectorState,
    runtime: Runtime[GraphConnectorContext],
) -> _GraphConnectorState:
    current_chunk = state.chunks[state.current_chunk_index]
    logger.info("Predicting TTPs for chunk %d/%d", state.current_chunk_index + 1, len(state.chunks))

    # Predict TTPs for chunk
    output = tactics_predictor_agent.invoke(
        TacticsPredictorInput(chunk=current_chunk["graph"].to_list()),
        context=TacticsPredictorContext(llm=runtime.context.llm),
    )
    logger.info("Predicted TTPs for chunk %d/%d", state.current_chunk_index + 1, len(state.chunks))

    # Assign TTPs to the current chunk
    current_chunk = current_chunk.with_columns(
        pl.Series("tactics", [output["tactics"]] * len(current_chunk)),
        pl.Series("techniques", [output["techniques"]] * len(current_chunk)),
    )

    new_chunks = state.chunks.copy()
    new_chunks[state.current_chunk_index] = current_chunk
    return replace(
        state,
        chunks=new_chunks,
    )


def _save_chunk(
    state: _GraphConnectorState,
    runtime: Runtime[GraphConnectorContext],
) -> _GraphConnectorState:
    chunk = state.chunks[state.current_chunk_index]
    for row in chunk.iter_rows(named=True):
        runtime.context.graph_store.add_graph(
            row["log event"],
            row["graph"],
            tactics=row["tactics"],
            techniques=row["techniques"],
        )
        runtime.context.vector_store.add_event(
            row["log event"],
        )
    logger.info("Saved chunk with %d graphs to store.", len(chunk))

    return replace(state, current_chunk_index=state.current_chunk_index + 1)


graph_connector_agent = StateGraph(
    _GraphConnectorState,
    input_schema=GraphConnectorInput,
    context_schema=GraphConnectorContext,
)


def _get_next_node(state: _GraphConnectorState) -> str:
    if state.current_chunk_index >= len(state.chunks):
        return END

    return BUILD_GRAPHS_FOR_CHUNK_NODE


graph_connector_agent.add_node(CHUNK_GRAPHS_NODE, _chunk_graphs)
graph_connector_agent.add_node(BUILD_GRAPHS_FOR_CHUNK_NODE, _build_graphs_for_chunk)
graph_connector_agent.add_node(PREDICT_TTPS_FOR_CHUNK_NODE, _predict_ttps_for_chunk)
graph_connector_agent.add_node(SAVE_CHUNK_NODE, _save_chunk)

graph_connector_agent.add_edge(START, CHUNK_GRAPHS_NODE)
graph_connector_agent.add_edge(CHUNK_GRAPHS_NODE, BUILD_GRAPHS_FOR_CHUNK_NODE)
graph_connector_agent.add_edge(BUILD_GRAPHS_FOR_CHUNK_NODE, PREDICT_TTPS_FOR_CHUNK_NODE)
graph_connector_agent.add_edge(PREDICT_TTPS_FOR_CHUNK_NODE, SAVE_CHUNK_NODE)
graph_connector_agent.add_conditional_edges(
    SAVE_CHUNK_NODE,
    _get_next_node,
    [END, BUILD_GRAPHS_FOR_CHUNK_NODE],
)


graph_connector_agent = graph_connector_agent.compile()
