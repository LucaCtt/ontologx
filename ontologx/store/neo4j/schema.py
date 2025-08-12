"""Internal schema management for the Neo4j store in OntoLogX."""

import uuid

from langchain_neo4j import Neo4jGraph


class Schema:
    """Schema management for the Neo4j store, following the MLSX ontology."""

    def __init__(self, graph_store: Neo4jGraph, study_uri: str, experiment_uri: str, run_uri: str) -> None:
        self.__graph_store = graph_store
        self.__study_uri = study_uri
        self.__experiment_uri = experiment_uri
        self.__run_uri = run_uri

    def initialize(self) -> None:
        """Initialize the graph store schema, by creating Study, Experiment, and Run nodes."""
        self.__initialize_study()
        self.__initialize_experiment()
        self.__initialize_run()

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
            MATCH (r:mlsx__Run {uri: $run_uri}), (m:mlsx__EvaluationMeasure {n4sch__name: $name})
            CREATE (r)-[:mlsx__hasOutput]->(e:mlsx__ModelEvaluation {mlsx__hasValue: $value})-[:mlsx__specifiedBy]->(m)
            """,
            params={
                "name": measure_camel,
                "value": evaluation,
                "run_uri": self.__run_uri,
            },
        )

    def add_hyperparameter(self, name: str, value: str | float | bool) -> None:
        """Add a hyperparameter to the run in the graph store.

        The hyperparameter is created if it does not exist, and it is attached to the run node.

        Args:
            name: The name of the hyperparameter.
            value: The value of the hyperparameter.

        """
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
            params={"name": name_camel, "value": value, "run_uri": self.__run_uri},
        )

    def __initialize_study(self) -> None:
        """Initialize the study node in the graph store.

        Only one study node is expected to exist in the graph store.
        If it does not exist, it will be created.
        """
        study = self.__graph_store.query(
            "MATCH (s:mlsx__Study {uri: $study_uri}) RETURN s",
            params={"study_uri": self.__study_uri},
        )
        if study:
            return

        # Create the study node if it does not exist
        self.__graph_store.query(
            "CREATE (s:mlsx__Study {uri: $uri, n4sch__name: 'OntoLogX'})",
            params={"uri": self.__study_uri},
        )

    def __initialize_experiment(self) -> None:
        """Initialize the experiment node in the graph store.

        The experiment node is created if it does not exist.
        """
        # Check if there is an experiment node with the same name under the given study
        exp = self.__graph_store.query(
            """
            MATCH (e:mlsx__Experiment {uri: $experiment_uri})<-[:mlsx__hasPart]-(s:mlsx__Study {uri: $study_uri})
            RETURN e
            LIMIT 1
            """,
            params={
                "experiment_uri": self.__experiment_uri,
                "study_uri": self.__study_uri,
            },
        )
        if exp:
            return

        # Create the experiment node if it does not exist
        self.__graph_store.query(
            """
            MATCH (s:mlsx__Study {uri: $study_uri})
            CREATE (e:mlsx__Experiment {uri: $experiment_uri})<-[:mlsx__hasPart]-(s)
            """,
            params={
                "experiment_uri": self.__experiment_uri,
                "study_uri": self.__study_uri,
            },
        )

    def __initialize_run(self) -> None:
        """Initialize the run node in the graph store.

        The run node is created if it does not exist, and it is attached to the experiment node.

        """
        # Create run
        self.__graph_store.query(
            """
            MATCH (e:mlsx__Experiment {uri: $experiment_uri})
            CREATE (r:mlsx__Run {uri: $run_uri})<-[:mlsx__hasPart]-(e)
            """,
            params={
                "run_uri": self.__run_uri,
                "experiment_uri": self.__experiment_uri,
            },
        )

    def __gen_uri(self) -> str:
        return f"{self.__run_uri}#{uuid.uuid4()!s}"
