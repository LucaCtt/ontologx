"""Main entry point for the log knowledge graph builder.

Handles the instantiation of the parser and the backend,
the reading of the logs, and the construction of the knowledge graph.
"""

import logging
import time

import typer
from langchain_neo4j.graphs.graph_document import GraphDocument
from neo4j.exceptions import ClientError
from rich.logging import RichHandler
from rich.progress import track

from ontologx import accuracy
from ontologx.backend import EmbeddingsFactory, LLMFactory
from ontologx.config import Config
from ontologx.parser import ParserFactory
from ontologx.store import Store

config = Config()


logger = logging.getLogger("rich")
logging.basicConfig(format="%(message)s", handlers=[RichHandler(omit_repeated_times=False)])
if config.debug:
    # Set up dev logging format
    logger.setLevel(logging.DEBUG)

# Load the embeddings model
embeddings = EmbeddingsFactory.create(
    backend=config.embeddings_backend,
    model=config.embeddings_model,
    url=config.embeddings_backend_url,
)

# Load the llm
llm = LLMFactory.create(
    backend=config.llm_backend,
    model=config.parser_model,
    temperature=config.parser_temperature,
    url=config.llm_backend_url,
)

# Create the vector store
store = Store(config=config, embeddings=embeddings)

app = typer.Typer()


@app.command()
def clear() -> None:
    store.clear()
    logger.info("Store cleared.")


@app.command()
def run() -> None:
    logger.info("Experiment: '%s'", config.experiment_name)
    logger.info("Embeddings model: '%s'", config.embeddings_model)
    logger.info("Language model: '%s'", config.parser_model)

    for _ in range(config.n_runs):
        config.new_run()
        logger.info("Run: '%s'", config.run_name)

        store.initialize()
        logger.info("Store at '%s' initialized.", config.neo4j_url)

        test_events = store.dataset.tests()
        logger.info("Read %d tests from '%s'", len(test_events), config.tests_path)

        parser = ParserFactory.create(
            config.parser_type,
            llm,
            store,
            config.prompt_build_graph,
            correction_steps=config.correction_steps,
        )

        total_time = 0
        total_success = 0
        total_shacl_violations = 0
        graphs_pred = []
        graphs_true = []

        for event, context, graph_true in track(test_events, description="Parsing events"):
            logger.debug("Parsing event: '%s'", event)
            start_time = time.time()

            graph_pred = parser.parse(event, context)
            total_time += time.time() - start_time

            if graph_pred is None:
                logger.warning("Event '%s' could not be parsed", event)
                # Add an empty graph to the list of predicted graphs
                # so the metrics will be calculated correctly
                graphs_pred.append(GraphDocument(nodes=[], relationships=[]))
            else:
                graphs_pred.append(graph_pred)
                total_success += 1

                try:
                    store.dataset.add_event_graph(graph_pred)
                except ClientError:
                    total_shacl_violations += 1

            graphs_true.append(graph_true)

        logger.info("Log parsing done.")

        avg_time = total_time / len(test_events)
        pct_success = total_success / len(test_events)
        pct_violations = total_shacl_violations / len(test_events)
        precision, recall, f1, ela, rla = accuracy.metrics(graphs_pred, graphs_true)

        store.schema.add_results(
            {
                "average_generation_time": avg_time,
                "generation_success_percentage": pct_success,
                "shacl_violations_percentage": pct_violations,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "entity_linking_accuracy": ela,
                "relationship_linking_accuracy": rla,
            },
        )

        logger.info("Average generation time: %f seconds", avg_time)
        logger.info("Success percentage: %f", pct_success)
        logger.info("SHACL violations percentage: %f", pct_violations)
        logger.info("Precision: %f", precision)
        logger.info("Recall: %f", recall)
        logger.info("F1 score: %f", f1)
        logger.info("Entity linking accuracy: %f", ela)
        logger.info("Relationship linking accuracy: %f", rla)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
