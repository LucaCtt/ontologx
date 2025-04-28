import uuid
from typing import Any

from langchain_neo4j import Neo4jGraph

from ontologx.config import Config
from ontologx.store.module import StoreModule

MLSCHEMA_URI = "https://raw.githubusercontent.com/ML-Schema/core/master/MLSchema.ttl"


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

        uri = self.__gen_uri()
        # Create the study node if it does not exist
        self.__graph_store.query("CREATE (s:Study {uri: $uri})", params={"uri": uri})

        return uri

    def __initialize_experiment(self, study_uri: str) -> str:
        """Initialize the experiment node in the graph store.

        The experiment node is created if it does not exist.

        Args:
            study_uri (str): The URI of the study node.

        Returns:
            str: The URI of the experiment node.

        """
        # Check if there is an experiment node with the same ontology and examples.
        exp = self.__graph_store.query(
            """
            MATCH (e:Experiment {name: $name})<-[:hasPart]-(s:Study {uri: $study_uri, name: 'OntoLogX'})
            RETURN e.uri as uri
            LIMIT 1
            """,
            params={
                "name": self.__config.experiment_name,
                "study_uri": study_uri,
            },
        )
        if exp:
            return exp[0]["uri"]

        # Create the experiment node if it does not exist
        uri = self.__gen_uri()
        self.__graph_store.query(
            """
            MATCH (s:Study {uri: $study_uri})
            CREATE (e:Experiment $details)<-[:hasPart]-(s)
            """,
            params={
                "details": {
                    "uri": uri,
                    "name": self.__config.experiment_name,
                },
                "study_uri": study_uri,
            },
        )

        return uri

    def __initialize_run(self, experiment_uri: str) -> None:
        # Create the run node and attach it to the experiment node
        run_uri = self.__gen_uri(self.__config.run_name)

        # Create run
        self.__graph_store.query(
            """
            MATCH (e:Experiment {uri: $experiment_uri})
            CREATE (r:Run $details)<-[:hasPart]-(e)
            """,
            params={
                "details": {"uri": run_uri, "runName": self.__config.run_name},
                "experiment_uri": experiment_uri,
            },
        )

        # Create hyperparameter nodes
        for name, value in self.__config.hyperparameters().items():
            name_camel = "".join(x.capitalize() for x in name.lower().split("_"))

            param = self.__graph_store.query(
                """
                MATCH (h:HyperParameter {name: $name})
                RETURN h
                LIMIT 1
                """,
                params={"name": name_camel},
            )

            if not param:
                self.__graph_store.query(
                    "CREATE (h:HyperParameter {name: $name, uri: $uri})",
                    params={"name": name_camel, "uri": self.__gen_uri()},
                )

            self.__graph_store.query(
                """
                MATCH (r:Run {uri: $run_uri}), (h:HyperParameter {name: $name})
                CREATE (r)-[:hasInput]->(s:HyperParameterSetting {hasValue: $value})-[:specifiedBy]->(h)
                """,
                params={"name": name_camel, "value": value, "run_uri": run_uri},
            )

    def __gen_uri(self, node_id: str = str(uuid.uuid4())) -> str:
        return f"{self.__config.run_uri}#{node_id}"
