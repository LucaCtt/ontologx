from pathlib import Path

from langchain_neo4j import Neo4jGraph

from ontologx.config import Config
from ontologx.store import GraphDocument, Node, Relationship, StoreModule

TIME_ONTOLOGY_URI = "http://www.w3.org/2006/time#"
MLSCHEMA_ONTOLOGY_URI = "https://www.w3.org/ns/mls#"
XML_SCHEMA_URI = "http://www.w3.org/2001/XMLSchema#"
OWL_SCHEMA_URI = "http://www.w3.org/2002/07/owl#"


class Ontology(StoreModule):
    """Ontology store module.

    This module is responsible for initializing the ontology store and
    providing access to the ontology.
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

        # Load the constraints
        self.__graph_store.query(
            "CALL n10s.validation.shacl.import.inline($constraints, 'Turtle')",
            params={"constraints": Path(self.__config.shacl_path).read_text()},
        )

        # Add transaction validator
        self.__graph_store.query(
            """
            USE system
            CALL apoc.trigger.install(
                'neo4j',
                $trigger_name,
                'call n10s.validation.shacl.validateTransaction(
                    $createdNodes,
                    $createdRelationships,
                    $assignedLabels,
                    $removedLabels,
                    $assignedNodeProperties,
                    $removedNodeProperties,
                    $deletedRelationships,
                    $deletedNodes)',
                { labels: ['Run']},
                {phase:'before'})
            """,
            params={"trigger_name": self.__config.n10s_trigger_name},
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
            WHERE c.uri STARTS WITH $log_ontology_uri OR c.uri = $time_instant_uri OR c.uri = $time_datetime_uri
            OPTIONAL MATCH (c)<-[:DOMAIN]-(p:Property)
            WITH c.name AS class, c.uri as uri, COLLECT([p.name, p.comment]) AS pairs
            RETURN class, uri, apoc.map.fromPairs(pairs) AS properties
            """,
            params={
                "log_ontology_uri": self.__config.ontology_uri,
                "time_instant_uri": f"{TIME_ONTOLOGY_URI}#Instant",
                "time_datetime_uri": f"{TIME_ONTOLOGY_URI}#GeneralDateTimeDescription",
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
                n.uri = $time_instant_uri
                AND m.uri = $time_datetime_uri
            )
            RETURN n.uri AS subject_uri, r.name AS predicate, m.uri AS object_uri
            """,
            params={
                "log_ontology_uri": self.__config.ontology_uri,
                "time_instant_uri": f"{TIME_ONTOLOGY_URI}#Instant",
                "time_datetime_uri": f"{TIME_ONTOLOGY_URI}#GeneralDateTimeDescription",
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
        )
