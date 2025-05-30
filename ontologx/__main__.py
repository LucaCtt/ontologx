"""Main entry point for the log knowledge graph builder.

Handles the instantiation of the parser and the backend,
the reading of the logs, and the construction of the knowledge graph.
"""

import logging
import time

import typer
from neo4j.exceptions import ClientError
from rich.logging import RichHandler
from rich.progress import track

from ontologx import accuracy
from ontologx.backend import EmbeddingsFactory, LLMFactory, TestsFactory
from ontologx.config import Config
from ontologx.parser import ParserFactory
from ontologx.store import GraphDocument
from ontologx.store.neo4j import Neo4jStore

config = Config()

# Setup logging
logging.basicConfig(
    format="%(message)s",
    handlers=[
        RichHandler(
            locals_max_string=200,
            tracebacks_code_width=200,
            tracebacks_width=None,
            omit_repeated_times=False,
            show_path=False,
        ),
    ],
)
logger = logging.getLogger("rich")
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
    model=config.llm_model,
    temperature=config.parser_temperature,
    url=config.llm_backend_url,
)

# Load the tests evaluator
tests_evaluator = TestsFactory.create(
    backend=config.tests_backend,
    model=config.tests_model,
    url=config.tests_backend_url,
)

# Create the vector store
store = Neo4jStore(config=config, embeddings=embeddings)

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def clear() -> None:
    store.clear()
    logger.info("Store cleared.")


@app.command()
def run() -> None:
    logger.info("Experiment: '%s'", config.experiment_name)
    logger.info("Embeddings model: '%s'", config.embeddings_model)
    logger.info("Language model: '%s'", config.llm_model)
    logger.info("Parser type: '%s'", config.parser_type)

    for _ in range(config.n_runs):
        config.new_run()
        logger.info("----------------------")
        logger.info("Run: '%s'", config.run_name)

        store.initialize()
        logger.info("Store at '%s' initialized.", config.neo4j_url)

        # Read the events at every run just in case,
        # to avoid leaking data between runs
        test_events = store.dataset.tests()
        logger.info("Read %d tests from '%s'", len(test_events), config.tests_path)

        parser = ParserFactory.create(
            config.parser_type,
            llm,
            store,
            config.prompt_build_graph,
            examples_retrieval=config.examples_retrieval,
            correction_steps=config.correction_steps,
        )
        logger.info("Parser '%s' created.", config.parser_type)

        total_time = 0
        total_success = 0
        total_shacl_violations = 0
        graphs_pred = []
        graphs_true = []

        for graph_true in track(test_events, description="Parsing events"):
            if graph_true.source is None:
                msg = "Test source is None. This is a bug."
                raise ValueError(msg)

            event = graph_true.source.page_content
            context = graph_true.source.metadata

            logger.info("Parsing event: '%s'", event)

            start_time = time.time()
            graph_pred = parser.parse(event, context)
            total_time += time.time() - start_time

            if graph_pred is None:
                logger.warning("Event '%s' could not be parsed.", event)
                # Add an empty graph to the list of predicted graphs
                # so the metrics will be calculated correctly
                graphs_pred.append(GraphDocument(nodes=[], relationships=[]))
            else:
                logger.info("Event parsed successfully.")
                graphs_pred.append(graph_pred)
                total_success += 1

                try:
                    store.dataset.add_event_graph(graph_pred)
                except ClientError:
                    total_shacl_violations += 1

            graphs_true.append(graph_true)

        logger.info("-------------------------")
        logger.info("Log parsing done.")

        metrics = accuracy.AccuracyEvaluator(graphs_pred, graphs_true, tests_evaluator)

        results = [
            ("total_run_time", total_time),
            ("average_generation_time", total_time / len(test_events)),
            ("generation_success_percentage", total_success / len(test_events)),
            ("SHACL_violations_percentage", total_shacl_violations / len(test_events)),
            ("precision", metrics.precision()),
            ("recall", metrics.recall()),
            ("f1_score", metrics.f1()),
            ("entity_linking_accuracy", metrics.entity_linking_accuracy()),
            ("relationship_linking_accuracy", metrics.relationship_linking_accuracy()),
            ("alignment", metrics.alignment()),
            ("BERT score", metrics.bert_score()),
        ]

        for name, value in results:
            logger.info("%s: %f", name.replace("_", " ").capitalize(), value)
            store.add_evaluation_result(name, value)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
