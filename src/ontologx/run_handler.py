"""Module to handle the run of the log knowledge graph builder."""

import logging
import time
import uuid
from pathlib import Path

from rich.progress import track

from ontologx.backend import EmbeddingsFactory, LLMFactory
from ontologx.config import Config
from ontologx.metrics import GEvalGraphAlignmentMetrics, OntologyMetrics, SHACLMetrics, TacticsMetrics
from ontologx.parser import ParserFactory
from ontologx.store import GraphDocument, Store
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
        self.__tests_model = LLMFactory.create(
            backend=config.geval_backend,
            model=config.geval_model,
            temperature=0.4,
            url=config.geval_backend_url,
        )
        self.__tactics_model = LLMFactory.create(
            backend=config.tactics_backend,
            model=config.tactics_model,
            temperature=0.4,
            url=config.tactics_backend_url,
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
        store = self.__initialize_new_store()
        logger.info("Store at '%s' initialized.", self.__config.neo4j_url)

        # Read the events at every run just in case,
        # to avoid leaking data between runs
        test_events = store.tests()
        logger.info("Read %d tests from '%s'", len(test_events), self.__config.tests_path)

        parser = ParserFactory.create(
            self.__config.parser_type,
            self.__parser_model,
            store,
            Path(self.__config.parser_prompt_path).read_text(),
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
            context = graph_true.source.metadata["context"]

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

                try:
                    store.add_event_graph(graph_pred)
                except Exception:
                    logger.exception("Error storing graph for event '%s'", event)

            graphs_true.append(graph_true)

        logger.info("-------------------------")
        logger.info("Log parsing done.")

        results = self.__compute_metrics(graphs_pred, graphs_true)
        results.update(
            {
                "run_total_time": total_time,
                "mean_generation_time": total_time / len(test_events),
                "generation_success_ratio": total_success / len(test_events),
            },
        )

        for name, value in results.items():
            logger.info("%s: %f", name.replace("_", " ").capitalize(), value)
            store.add_evaluation_result(name, value)

        for key, value in self.__config.__dict__.items():
            if key.startswith("neo4j"):
                continue

            store.add_hyperparameter(key, value)

    def __initialize_new_store(self) -> Store:
        run_uri = f"{self.__experiment_uri}/{uuid.uuid4()!s}"

        config = StoreConfig(
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
        store = Neo4jStore(self.__embeddings, config)
        store.initialize()

        return store

    def __compute_metrics(
        self,
        y_pred: list[GraphDocument],
        y_true: list[GraphDocument],
    ) -> dict[str, float]:
        """Compute the metrics for the given predicted and true graphs.

        Args:
            y_pred (list[GraphDocument]): The list of predicted graphs.
            y_true (list[GraphDocument]): The list of true graphs.

        Returns:
            dict: A dictionary containing the computed metrics.

        """
        results = {}

        if "ontology" in self.__config.metrics:
            ontology_metrics = OntologyMetrics(
                y_pred,
                y_true,
            )
            results.update(
                {
                    "precision": ontology_metrics.precision,
                    "recall": ontology_metrics.recall,
                    "f1_score": ontology_metrics.f1,
                    "entity_linking_accuracy": ontology_metrics.entity_linking_accuracy,
                    "relationship_linking_accuracy": ontology_metrics.relationship_linking_accuracy,
                },
            )

        if "shacl" in self.__config.metrics or "g-eval" in self.__config.metrics:
            shacl_metrics = SHACLMetrics(
                y_pred,
                self.__config.ontology_path,
                self.__config.shacl_path,
            )
            results.update(
                {
                    "SHACL_violations_ratio": shacl_metrics.violations_ratio,
                },
            )

            if "g-eval" in self.__config.metrics:
                geval_metrics = GEvalGraphAlignmentMetrics(
                    y_pred,
                    shacl_metrics.compliance_list,
                    self.__tests_model,
                )
                results.update(
                    {
                        "g-eval_mean_all": geval_metrics.mean,
                        "g-eval_mean_with_compliance": geval_metrics.mean_with_compliance,
                    },
                )

        if "tactics" in self.__config.metrics:
            ttps_metrics = TacticsMetrics(y_pred, y_true, self.__tactics_model, self.__config.tactics_prompt_path)
            results.update(
                {
                    "tactics_precision": ttps_metrics.precision,
                    "tactics_recall": ttps_metrics.recall,
                    "tactics_f1_score": ttps_metrics.f1_score,
                },
            )

        return results
