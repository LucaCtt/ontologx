"""Internal schema management for the Neo4j store in OntoLogX."""

import uuid

from langchain_neo4j import Neo4jGraph

from ontologx.config import Config


class Schema:
    """Graph store and vector index for the events knowledge graph."""

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

        The ontology and examples data is not experiment-tagged.
        To reset it, the store must be cleared and re-initialized.
        This is intentional, as the ontology and examples are expected to be static among experiments.
        """
        study_uri = self.__initialize_study()
        experiment_uri = self.__initialize_experiment(study_uri)
        self.__initialize_run(experiment_uri)

    def add_evaluation_result(self, measure: str, evaluation: str | float) -> None:
        """Add the evaluation result to the graph store."""
        measure_camel = "".join(x.capitalize() for x in measure.lower().split("_"))

        # Check if measure already exists
        measure_node = self.__graph_store.query(
            """
            MATCH (m:mlsx__EvaluationMeasure {n4sch__name: $name})
            RETURN m
            LIMIT 1
            """,
            params={"name": measure_camel},
        )
        if not measure_node:
            # Create the measure node if it does not exist
            self.__graph_store.query(
                """
                CREATE (m:mlsx__EvaluationMeasure {n4sch__name: $name, uri: $uri})
                """,
                params={"name": measure_camel, "uri": self.__gen_uri()},
            )

        # Create the evaluation node
        self.__graph_store.query(
            """
            MATCH (r:mlsx__Run {n4sch__runName: $run_name}), (m:mlsx__EvaluationMeasure {n4sch__name: $name})
            CREATE (r)-[:mlsx__hasOutput]->(e:mlsx__ModelEvaluation {mlsx__hasValue: $value})-[:mlsx__specifiedBy]->(m)
            """,
            params={
                "name": measure_camel,
                "value": evaluation,
                "run_name": self.__config.run_name,
            },
        )

    def __initialize_study(self) -> str:
        """Initialize the study node in the graph store.

        Only one study node is expected to exist in the graph store.
        If it does not exist, it will be created.

        Returns:
            str: The URI of the study node.

        """
        study = self.__graph_store.query(
            "MATCH (s:mlsx__Study) RETURN s.uri as uri LIMIT 1",
        )
        if study:
            return study[0]["uri"]

        # Create the study node if it does not exist
        uri = self.__gen_uri()
        self.__graph_store.query("CREATE (s:mlsx__Study {uri: $uri, n4sch__name: 'OntoLogX'})", params={"uri": uri})

        return uri

    def __initialize_experiment(self, study_uri: str) -> str:
        """Initialize the experiment node in the graph store.

        The experiment node is created if it does not exist.

        Args:
            study_uri (str): The URI of the study node.

        Returns:
            str: The URI of the experiment node.

        """
        # Check if there is an experiment node with the same name under the given study
        exp = self.__graph_store.query(
            """
            MATCH (e:mlsx__Experiment)<-[:mlsx__hasPart]-(s:mlsx__Study {uri: $study_uri})
            WHERE e.n4sch__name = $name
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
            MATCH (s:mlsx__Study {uri: $study_uri})
            CREATE (e:mlsx__Experiment $details)<-[:mlsx__hasPart]-(s)
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
        """Initialize the run node in the graph store.

        The run node is created if it does not exist, and it is attached to the experiment node.

        Args:
            experiment_uri (str): The URI of the experiment node.

        """
        # Create the run node and attach it to the experiment node
        run_uri = self.__gen_uri(self.__config.run_name)

        # Create run
        self.__graph_store.query(
            """
            MATCH (e:mlsx__Experiment {uri: $experiment_uri})
            CREATE (r:mlsx__Run $details)<-[:mlsx__hasPart]-(e)
            """,
            params={
                "details": {"uri": run_uri, "n4sch__runName": self.__config.run_name},
                "experiment_uri": experiment_uri,
            },
        )

        # Create hyperparameter nodes
        for name, value in self.__config.hyperparameters().items():
            name_camel = "".join(x.capitalize() for x in name.lower().split("_"))

            param = self.__graph_store.query(
                """
                MATCH (h:mlsx__HyperParameter {n4sch__name: $name})
                RETURN h
                LIMIT 1
                """,
                params={"name": name_camel},
            )

            if not param:
                self.__graph_store.query(
                    "CREATE (h:mlsx__HyperParameter {n4sch__name: $name, uri: $uri})",
                    params={"name": name_camel, "uri": self.__gen_uri()},
                )

            self.__graph_store.query(
                """
                MATCH (r:mlsx__Run {uri: $run_uri}), (h:mlsx__HyperParameter {n4sch__name: $name})
                CREATE (r)-[:mlsx__hasInput]->(s:mlsx__HyperParameterSetting {mlsx__hasValue: $value})
                -[:mlsx__specifiedBy]->(h)
                """,
                params={"name": name_camel, "value": value, "run_uri": run_uri},
            )

    def __gen_uri(self, node_id: str = str(uuid.uuid4())) -> str:
        return f"{self.__config.run_uri}#{node_id}"
