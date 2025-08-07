"""Module to handle the run of the log knowledge graph builder."""

import logging
import time
import uuid
from pathlib import Path

from rich.progress import track

from ontologx.backend import EmbeddingsFactory, LLMFactory, TestsFactory
from ontologx.config import Config
from ontologx.metrics.metrics import MetricsEvaluator
from ontologx.parser import ParserFactory
from ontologx.store import GraphDocument
from ontologx.store.config import StoreAuth, StoreConfig
from ontologx.store.neo4j.neo4j_store import Neo4jStore

_BASE_URI = "https://cyberseclab.unibs.it/olx"

logger = logging.getLogger("rich")


class RunHandler:
    """Handler for running the log knowledge graph builder."""

    def __init__(self, config: Config):
        self.__config = config

        self.__embeddings = EmbeddingsFactory.create(
            backend=config.embeddings_backend,
            model=config.embeddings_model,
            url=config.embeddings_backend_url,
        )
        self.__parser_model = LLMFactory.create(
            backend=config.parser_backend,
            model=config.parser_model,
            temperature=config.parser_temperature,
            url=config.parser_backend_url,
        )
        self.__tests_model = TestsFactory.create(
            backend=config.tests_backend,
            model=config.tests_model,
            url=config.tests_backend_url,
        )

        self.__study_uri = f"{_BASE_URI}/study"
        self.__experiment_uri = f"{self.__study_uri}/{config.experiment_name}"

    def start_new_run(self) -> None:
        """Start a run of the log knowledge graph builder.

        Returns:
            A tuple containing two lists:
            - Predicted graphs (GraphDocument)
            - True graphs (GraphDocument)

        """
        store = Neo4jStore(self.__embeddings, self.__get_store_config())
        store.initialize()

        logger.info("Store at '%s' initialized.", self.__config.neo4j_url)

        # Read the events at every run just in case,
        # to avoid leaking data between runs
        test_events = store.tests()
        logger.info("Read %d tests from '%s'", len(test_events), self.__config.tests_path)

        parser = ParserFactory.create(
            self.__config.parser_type,
            self.__parser_model,
            store,
            Path(self.__config.prompt_path).read_text(),
            examples_retrieval=self.__config.examples_retrieval,
            correction_steps=self.__config.correction_steps,
        )
        logger.info("Parser '%s' created.", self.__config.parser_type)

        total_time = 0
        total_success = 0
        graphs_pred = []
        graphs_true = []

        for graph_true in track(test_events, description="Parsing events"):
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
                graphs_pred.append(GraphDocument(nodes=[], relationships=[], source=graph_true.source))
            else:
                logger.info("Event parsed successfully.")
                graphs_pred.append(graph_pred)
                total_success += 1

                store.add_event_graph(graph_pred)

            graphs_true.append(graph_true)

        logger.info("-------------------------")
        logger.info("Log parsing done.")

        metrics = MetricsEvaluator(
            graphs_pred,
            graphs_true,
            self.__tests_model,
            self.__config.ontology_path,
            self.__config.shacl_path,
        )

        results = [
            ("run_total_time", total_time),
            ("mean_generation_time", total_time / len(test_events)),
            ("generation_success_ratio", total_success / len(test_events)),
            ("SHACL_violations_ratio", metrics.shacl_violations_ratio),
            ("precision", metrics.precision),
            ("recall", metrics.recall),
            ("f1_score", metrics.f1),
            ("entity_linking_accuracy", metrics.entity_linking_accuracy),
            ("relationship_linking_accuracy", metrics.relationship_linking_accuracy),
            ("g-eval_mean_all", metrics.geval_mean),
            ("g-eval_mean_with_compliance", metrics.geval_mean_with_compliance),
        ]

        for name, value in results:
            logger.info("%s: %f", name.replace("_", " ").capitalize(), value)
            store.add_evaluation_result(name, value)

        for key, value in self.__config.__dict__.items():
            if key.startswith("neo4j"):
                continue

            if isinstance(value, str | int | float | bool):
                store.add_hyperparameter(key, value)

        store.close()

    def __get_store_config(self) -> StoreConfig:
        run_uri = f"{self.__experiment_uri}/{uuid.uuid4()!s}"

        return StoreConfig(
            study_uri=self.__study_uri,
            experiment_uri=self.__experiment_uri,
            run_uri=run_uri,
            ontology_path=self.__config.ontology_path,
            examples_path=self.__config.examples_path,
            tests_path=self.__config.tests_path,
            generated_graphs_retrieval=self.__config.generated_graphs_retrieval,
            auth=StoreAuth(
                url=self.__config.neo4j_url,
                username=self.__config.neo4j_username,
                password=self.__config.neo4j_password,
            ),
        )
