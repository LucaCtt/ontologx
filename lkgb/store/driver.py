from typing import Any

from langchain_neo4j import Neo4jGraph

from lkgb.config import Config
from lkgb.store.module import StoreModule


class Driver(StoreModule):
    """Graph store and vector index for the events knowledge graph.

    This class uses LangChain's Neo4jGraph api. It does not use the Neo4jVector api
    because it is not flexible enough.
    """

    def __init__(
        self,
        config: Config,
    ) -> None:
        super().__init__(config)
        self.__graph_store = Neo4jGraph(
            url=config.neo4j_url,
            username=config.neo4j_username,
            password=config.neo4j_password,
        )

    def initialize(self) -> None:
        """Initialize the graph store and the vector index.

        The graph store is initialized with the ontology and the labelled examples.
        The vector index is created for the event nodes, also populating the embeddings.
        Note: the ontology and examples data is not experiment-tagged.
        To reset it, the store must be cleared and re-initialized.
        This is intentional, as the ontology and examples are expected to be static among experiments.
        """
        # Get the latest experiment node
        latest_experiment = self.__graph_store.query(
            """MATCH (n:Experiment)
            RETURN elementID(n) as id,
                n.experiment_date_time as experimentDateTime,
                n.ontology_hash as ontologyHash,
                n.examples_hash as examplesHash
            ORDER BY experimentDateTime DESC
            LIMIT 1
            """,
        )
        if latest_experiment:
            if latest_experiment[0]["ontologyHash"] != self._config.ontology_hash():
                msg = "The ontology has changed since the last experiment."
                raise ValueError(msg)

            if latest_experiment[0]["examplesHash"] != self._config.examples_hash():
                msg = "The examples have changed since the last experiment."
                raise ValueError(msg)

            self.__graph_store.query(
                """
                MATCH (m:Experiment)
                WHERE elementID(m) = $id
                CREATE (n:Experiment $details)-[:SUBSEQUENT]->(m)
                """,
                params={"details": self._config.dump(), "id": latest_experiment[0]["id"]},
            )
        else:
            # Create the experiment node
            self.__graph_store.query(
                """
                CREATE (n:Experiment $details)
                """,
                params={"details": self._config.dump()},
            )

    def query(self, query: str, params: dict | None = None) -> list[dict[str, Any]]:
        if params is None:
            params = {}
        return self.__graph_store.query(query, params)

    def clear(self) -> None:
        """Clear any experiment in the graph store."""
        self.__graph_store.query("MATCH (n:Experiment) DETACH DELETE n")
