"""Internal schema management for the Neo4j store in OntoLogX."""

import uuid

from neo4j import Driver


class Schema:
    """Schema management for the Neo4j store, following the MLSX ontology."""

    def __init__(self, driver: Driver, study_uri: str, experiment_uri: str, run_uri: str) -> None:
        self.__driver = driver
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
        measure_node, _, _ = self.__driver.execute_query(
            """
            MATCH (m:mlsx__EvaluationMeasure {n4sch__name: $name})
            RETURN m
            LIMIT 1
            """,
            name=measure_camel,
        )
        if not measure_node:
            # Create the measure node if it does not exist
            self.__driver.execute_query(
                """
                CREATE (m:mlsx__EvaluationMeasure {n4sch__name: $name, uri: $uri})
                """,
                name=measure_camel,
                uri=self.__gen_uri(),
            )

        # Create the evaluation node
        self.__driver.execute_query(
            """
            MATCH (r:mlsx__Run {uri: $run_uri}), (m:mlsx__EvaluationMeasure {n4sch__name: $name})
            CREATE (r)-[:mlsx__hasOutput]->(e:mlsx__ModelEvaluation {mlsx__hasValue: $value})-[:mlsx__specifiedBy]->(m)
            """,
            name=measure_camel,
            value=evaluation,
            run_uri=self.__run_uri,
        )

    def add_hyperparameter(self, name: str, value: str | float | bool) -> None:
        """Add a hyperparameter to the run in the graph store.

        The hyperparameter is created if it does not exist, and it is attached to the run node.

        Args:
            name: The name of the hyperparameter.
            value: The value of the hyperparameter.

        """
        name_camel = "".join(x.capitalize() for x in name.lower().split("_"))

        param, _, _ = self.__driver.execute_query(
            """
            MATCH (h:mlsx__HyperParameter {n4sch__name: $name})
            RETURN h
            LIMIT 1
            """,
            name=name_camel,
        )

        if not param:
            self.__driver.execute_query(
                "CREATE (h:mlsx__HyperParameter {n4sch__name: $name, uri: $uri})",
                name=name_camel,
                uri=self.__gen_uri(),
            )

        self.__driver.execute_query(
            """
            MATCH (r:mlsx__Run {uri: $run_uri}), (h:mlsx__HyperParameter {n4sch__name: $name})
            CREATE (r)-[:mlsx__hasInput]->(s:mlsx__HyperParameterSetting {mlsx__hasValue: $value})
            -[:mlsx__specifiedBy]->(h)
            """,
            name=name_camel,
            value=value,
            run_uri=self.__run_uri,
        )

    def __initialize_study(self) -> None:
        """Initialize the study node in the graph store.

        Only one study node is expected to exist in the graph store.
        If it does not exist, it will be created.
        """
        study, _, _ = self.__driver.execute_query(
            "MATCH (s:mlsx__Study {uri: $study_uri}) RETURN s",
            study_uri=self.__study_uri,
        )
        if study:
            return

        # Create the study node if it does not exist
        self.__driver.execute_query(
            "CREATE (s:mlsx__Study {uri: $uri, n4sch__name: 'OntoLogX'})",
            uri=self.__study_uri,
        )

    def __initialize_experiment(self) -> None:
        """Initialize the experiment node in the graph store.

        The experiment node is created if it does not exist.
        """
        # Check if there is an experiment node with the same name under the given study
        exp, _, _ = self.__driver.execute_query(
            """
            MATCH (e:mlsx__Experiment {uri: $experiment_uri})<-[:mlsx__hasPart]-(s:mlsx__Study {uri: $study_uri})
            RETURN e
            LIMIT 1
            """,
            experiment_uri=self.__experiment_uri,
            study_uri=self.__study_uri,
        )
        if exp:
            return

        # Create the experiment node if it does not exist
        self.__driver.execute_query(
            """
            MATCH (s:mlsx__Study {uri: $study_uri})
            CREATE (e:mlsx__Experiment {uri: $experiment_uri})<-[:mlsx__hasPart]-(s)
            """,
            experiment_uri=self.__experiment_uri,
            study_uri=self.__study_uri,
        )

    def __initialize_run(self) -> None:
        """Initialize the run node in the graph store.

        The run node is created if it does not exist, and it is attached to the experiment node.

        """
        # Create run
        self.__driver.execute_query(
            """
            MATCH (e:mlsx__Experiment {uri: $experiment_uri})
            CREATE (r:mls__Run {uri: $run_uri})<-[:mls__hasPart]-(e)
            """,
            run_uri=self.__run_uri,
            experiment_uri=self.__experiment_uri,
        )

    def __gen_uri(self) -> str:
        return f"{self.__run_uri}#{uuid.uuid4()!s}"
