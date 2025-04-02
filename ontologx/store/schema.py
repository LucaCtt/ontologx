import uuid
from typing import Any

from langchain_neo4j import Neo4jGraph

from ontologx.config import Config
from ontologx.store.module import StoreModule


class Schema(StoreModule):
    """Graph store and vector index for the events knowledge graph.

    This class uses LangChain's Neo4jGraph api. It does not use the Neo4jVector api
    because it is not flexible enough.
    """

    def __init__(
        self,
        config: Config,
        graph_store: Neo4jGraph,
    ) -> None:
        self.__config = config
        self.__graph_store = graph_store

    def initialize(self) -> None:
        """Initialize the graph store and the vector index.

        The graph store is initialized with the ontology and the labelled examples.
        The vector index is created for the event nodes, also populating the embeddings.
        Note: the ontology and examples data is not experiment-tagged.
        To reset it, the store must be cleared and re-initialized.
        This is intentional, as the ontology and examples are expected to be static among experiments.
        """
        # Create the study node if it does not exist
        study_uri = self.__initialize_study()
        experiment_uri = self.__initialize_experiment(study_uri)
        self.__initialize_run(experiment_uri)

    def query(self, query: str, params: dict | None = None) -> list[dict[str, Any]]:
        if params is None:
            params = {}
        return self.__graph_store.query(query, params)

    def clear(self) -> None:
        """Clear any experiment in the graph store."""
        self.__graph_store.query("MATCH (s:Study) DETACH DELETE s")
        self.__graph_store.query("MATCH (e:Experiment) DETACH DELETE e")
        self.__graph_store.query("MATCH (r:Run) DETACH DELETE r")

    def __initialize_study(self) -> str:
        """Initialize the study node in the graph store.

        The study node is created if it does not exist.

        Returns:
            str: The URI of the study node.

        """
        study = self.__graph_store.query(
            """
            MATCH (s:Study)
            RETURN s.uri as uri
            LIMIT 1
            """,
        )
        if study:
            return study[0]["uri"]

        uri = self.gen_uri()
        # Create the study node if it does not exist
        self.__graph_store.query("MERGE (s:Study {uri: $uri})", params={"uri": uri})

        return uri

    def __initialize_experiment(self, study_uri: str) -> str:
        """Initialize the experiment node in the graph store.

        The experiment node is created if it does not exist.

        Args:
            study_uri (str): The URI of the study node.

        Returns:
            str: The URI of the experiment node.

        """
        ontology_hash = self.__config.ontology_hash()
        examples_hash = self.__config.examples_hash()
        tests_hash = self.__config.tests_hash()

        # Check if there is an experiment node with the same ontology and examples.
        exp = self.__graph_store.query(
            """
            MATCH (e:Experiment $details)<-[:hasPart]-(s:Study {uri: $study_uri})
            RETURN e.uri as uri
            LIMIT 1
            """,
            params={
                "details": {"ontologyHash": ontology_hash, "examplesHash": examples_hash, "testsHash": tests_hash},
                "study_uri": study_uri,
            },
        )
        if exp:
            return exp[0]["uri"]

        # Create the experiment node if it does not exist
        uri = self.gen_uri()
        self.__graph_store.query(
            """
            CREATE (e:Experiment $details)<-[:hasPart]-(s:Study {uri: $study_uri})
            """,
            params={
                "details": {
                    "uri": uri,
                    "dateTime": self.__config.experiment_date_time,
                    "ontologyHash": ontology_hash,
                    "examplesHash": examples_hash,
                    "testsHash": tests_hash,
                },
                "study_uri": study_uri,
            },
        )

        return uri

    def __initialize_run(self, experiment_uri: str) -> None:
        # Create the run node and attach it to the experiment node
        self.__graph_store.query(
            """
            CREATE (r:Run $details)<-[:hasPart]-(e:Experiment {uri: $experiment_uri})
            """,
            params={
                "details": {"uri": self.gen_uri(self.__config.run_id)},
                "experiment_uri": experiment_uri,
            },
        )

        for name, value in self.__config.hyperparameters().items():
            param = self.__graph_store.query(
                """
                MATCH (h:HyperParameter $details)
                WHERE h.hasName = $name
                RETURN h
                LIMIT 1
                """,
            )

            if not param:
                self.__graph_store.query(
                    "CREATE (h:HyperParameter {hasName: $name})",
                    params={"name": name, "uri": self.gen_uri()},
                )

            self.__graph_store.query(
                """
                MATCH (s:HyperParameterSetting {hasValue: $value})-[:specifiedBy]->(h:HyperParameter {hasName: $name})
                """,
                params={"name": name, "value": value},
            )

    def gen_uri(self, node_id: str = str(uuid.uuid4())) -> str:
        return f"{self.__config.run_uri}#{node_id}"
