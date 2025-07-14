from pathlib import Path

from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph

from ontologx.config import Config
from ontologx.store import GraphDocument, Node, Relationship

TIME_ONTOLOGY_URI = "http://www.w3.org/2006/time#"
INSTANT_CLASS_URI = f"{TIME_ONTOLOGY_URI}Instant"
MLSCHEMA_ONTOLOGY_URI = "http://www.w3.org/ns/mls#"
XML_SCHEMA_URI = "http://www.w3.org/2001/XMLSchema#"
OWL_SCHEMA_URI = "http://www.w3.org/2002/07/owl#"


class Ontology:
    """Ontology store.

    Initializes the ontology on the store and provides access to the ontology.
    """

    def __init__(self, config: Config, graph_store: Neo4jGraph) -> None:
        self.__config = config
        self.__graph_store = graph_store

    def initialize(self) -> None:
        # Check if the neosemantics configuration is present,
        # if it is, assume the ontology is already loaded.
        result = self.__graph_store.query("MATCH (n:_GraphConfig) RETURN COUNT(n) AS count")
        if result[0]["count"] != 0:
            return

        # Init neosemantics plugin
        self.__graph_store.query("CALL n10s.graphconfig.init()")
        self.__graph_store.query("CALL n10s.graphconfig.set({ handleVocabUris: 'IGNORE' })")
        self.__graph_store.query(
            f"""CREATE CONSTRAINT {self.__config.n10s_constraint_name} IF NOT EXISTS
            FOR (r:Resource) REQUIRE r.uri IS UNIQUE""",
        )

        # Load the ontologies
        self.__graph_store.query(
            "CALL n10s.onto.import.inline($ontology, 'Turtle')",
            params={"ontology": Path(self.__config.ontology_path).read_text()},
        )
        self.__graph_store.query(
            "CALL n10s.onto.import.fetch($url, 'Turtle')",
            params={"url": TIME_ONTOLOGY_URI},
        )
        self.__graph_store.query(
            "CALL n10s.onto.import.fetch($url, 'Turtle')",
            params={"url": MLSCHEMA_ONTOLOGY_URI},
        )

        # Load the SHACL constraints
        self.__graph_store.query(
            "CALL n10s.validation.shacl.import.inline($constraints, 'Turtle')",
            params={"constraints": Path(self.__config.shacl_path).read_text()},
        )

    def graph(self) -> GraphDocument:
        """Return the ontology graph as a GraphDocument.

        The returned nodes and relationship types will be without uris. This may not be the best idea,
        only time will tell.

        Note that this will not return all of the classes and relationships from external ontologies,
        but only the relevant ones for this project.

        Returns:
            GraphDocument: The ontology graph, where nodes are classes
            and relationships are relationships between classes.

        """
        nodes_with_props = self.__graph_store.query(
            """
            MATCH (c:Class)
            WHERE c.uri STARTS WITH $log_ontology_uri OR c.uri = $time_instant_uri
            OPTIONAL MATCH (c)<-[:DOMAIN]-(p:Property)
            WITH c.name AS class, c.uri as uri, COLLECT([p.name, p.comment]) AS pairs
            RETURN class, uri, apoc.map.fromPairs(pairs) AS properties
            """,
            params={
                "log_ontology_uri": self.__config.ontology_uri,
                "time_instant_uri": INSTANT_CLASS_URI,
            },
        )
        nodes_dict = {
            row["uri"]: Node(id=row["uri"], type=row["class"], properties=row["properties"]) for row in nodes_with_props
        }

        triples = self.__graph_store.query(
            """
            MATCH (n:Class)<-[:DOMAIN]-(r:Relationship)-[:RANGE]->(m:Class)
            WHERE
            (
                n.uri STARTS WITH $log_ontology_uri
                AND m.uri STARTS WITH $log_ontology_uri
                AND r.uri STARTS WITH $log_ontology_uri
            )
            OR
            (
                n.uri STARTS WITH $log_ontology_uri
                and m.uri = $time_instant_uri
            )
            RETURN n.uri AS subject_uri, r.name AS predicate, m.uri AS object_uri
            """,
            params={
                "log_ontology_uri": self.__config.ontology_uri,
                "time_instant_uri": INSTANT_CLASS_URI,
            },
        )
        relationships = [
            Relationship(
                source=nodes_dict[row["subject_uri"]],
                target=nodes_dict[row["object_uri"]],
                type=row["predicate"],
            )
            for row in triples
        ]

        return GraphDocument(
            nodes=list(nodes_dict.values()),
            relationships=relationships,
            source=Document(
                page_content="Ontology graph",
                metadata={"ontology_uri": self.__config.ontology_uri},
            ),
        )

    def total_constraints(self) -> int:
        """Return the total number of SHACL constraints in the ontology.

        Returns:
            int: The total number of SHACL constraints.

        """
        result = self.__graph_store.query(
            "CALL n10s.validation.shacl.listShapes() YIELD target RETURN COUNT(target) AS count",
        )
        return result[0]["count"] if result else 0

    def validate(self, graph: GraphDocument) -> int:
        """Validate the given graph against the SHACL constraints.

        The nodes in the graph must already be present in the store.

        Args:
            graph (GraphDocument): The graph to validate.

        Returns:
            int: The number of validation errors found in the graph.

        """
        nodes_uris = [node.id for node in graph.nodes]

        result = self.__graph_store.query(
            """
            MATCH (n)
            WHERE n.uri IN $uris
            WITH COLLECT(n) AS nodes
            CALL n10s.validation.shacl.validateSet(nodes)
            YIELD focusNode, nodeType, propertyShape, offendingValue, resultPath, severity
            WHERE resultPath <> "uri" AND resultPath <> "embedding" AND resultPath <> "runName"
            RETURN focusNode, nodeType, propertyShape, offendingValue, resultPath, severity
            """,
            params={"uris": nodes_uris},
        )

        return len(result)
