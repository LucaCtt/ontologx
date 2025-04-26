"""Main entry point for the log knowledge graph builder.

Handles the instantiation of the parser and the backend,
the reading of the logs, and the construction of the knowledge graph.
"""

import logging
import time

import typer
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table

from ontologx import accuracy
from ontologx.backend import BackendFactory
from ontologx.config import Config
from ontologx.parser import Parser
from ontologx.store import Store

config = Config()

# Set up logging format
logging.basicConfig(format="%(message)s", handlers=[RichHandler(omit_repeated_times=False)])
logger = logging.getLogger("rich")
logger.setLevel(logging.DEBUG)

# Set the backend
backend = BackendFactory.create(config.backend)

# Load the embeddings model
embeddings = backend.embeddings(model=config.embeddings_model)

# Create the vector store
store = Store(config=config, embeddings=embeddings)

app = typer.Typer()


@app.command()
def clear() -> None:
    store.clear()
    logger.info("Store cleared.")


@app.command()
def parse() -> None:
    logger.info("Experiment: %s", config.experiment_name)
    logger.info("Run: %s", config.run_name)
    logger.info("Backend: %s", config.backend)
    logger.info("Embeddings model: '%s'", config.embeddings_model)
    logger.info("Language model: '%s'", config.parser_model)

    # Load the llm
    llm = backend.llm(model=config.parser_model, temperature=config.parser_temperature)

    store.initialize()
    logger.info("Store at '%s' initialized.", config.neo4j_url)

    test_events = store.dataset.tests()
    logger.info("Read %d tests from '%s'", len(test_events), config.tests_path)

    parser = Parser(llm, store, config.prompt_build_graph, config.self_reflection_steps)

    total_time = 0
    total_success = 0
    graphs_pred = []
    graphs_true = []

    for event, context, graph_true in track(test_events, description="Parsing events"):
        logger.debug("Parsing event: '%s'", event)
        start_time = time.time()

        graph_pred = parser.parse(event, context)
        total_time += (time.time() - start_time) / len(test_events)
        graphs_pred.append(graph_pred)
        graphs_true.append(graph_true)

        if graph_pred is None:
            logger.warning("Event '%s' could not be parsed", event)
        else:
            store.dataset.add_event_graph(graph_pred)
            total_success += 1

    logger.info("Log parsing done.")

    avg_time = total_time / len(test_events)
    pct_success = total_success / len(test_events)
    precision, recall, f1, ela, rla = accuracy.metrics(graphs_pred, graphs_true)

    table = Table("Metric", "Value", title="Run Summary")
    table.add_row("Experiment", config.experiment_name)
    table.add_row("Run", config.run_name)
    table.add_row("Backend", config.backend)
    table.add_row("Embeddings model", config.embeddings_model)
    table.add_row("Language model", config.parser_model)
    table.add_row("Events parsed", str(len(test_events)), end_section=True)
    table.add_row("Average generation time", f"{avg_time:.2f} seconds")
    table.add_row("Success percentage", f"{pct_success:.2%}")
    table.add_row("Precision", f"{precision:.2%}")
    table.add_row("Recall", f"{recall:.2%}")
    table.add_row("F1 score", f"{f1:.2%}")
    table.add_row("Entity linking accuracy", f"{ela:.2%}")
    table.add_row("Relationship linking accuracy", f"{rla:.2%}")

    logger.info(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
