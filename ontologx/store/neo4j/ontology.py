"""Ontology store for managing and querying ontologies in a Neo4j database."""

import functools
from pathlib import Path

from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph

from ontologx.store import GraphDocument, Node, Relationship
from ontologx.store.neo4j.utils import get_uri_from_ttl, normalize_output_graph

_TIME_ONTOLOGY_URI = "http://www.w3.org/2006/time#"
_MLSX_ONTOLOGY_URI = "https://cyberseclab.unibs.it/mlsx/dict#"
_ONTOLOGY_PARAMS = {
    "subClassOfRel": "subClassOf",
    "subPropertyOfRel": "subPropertyOf",
    "domainRel": "domain",
    "rangeRel": "range",
}


class Ontology:
    """Ontology store.

    Initializes the olx ontology on the store and provides access to the ontology.
    """

    def __init__(self, graph_store: Neo4jGraph, ontology_path: str) -> None:
        self.__graph_store = graph_store
        self.__ontology_path = ontology_path

    def initialize(self) -> None:
        """Import the olx ontology (and dependencies) into the store."""
        # Check if the neosemantics configuration is present,
        # if it is, assume the ontology is already loaded.
        result = self.__graph_store.query("MATCH (n:_GraphConfig) RETURN COUNT(n) AS count")
        if result[0]["count"] != 0:
            return

        # Init neosemantics plugin
        self.__graph_store.query("CALL n10s.graphconfig.init($params)", params={"params": _ONTOLOGY_PARAMS})
        self.__graph_store.query(
            "CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS FOR (r:Resource) REQUIRE r.uri IS UNIQUE",
        )

        # Set the namespaces
        self.__graph_store.query("CALL n10s.nsprefixes.add('olx', $uri)", params={"uri": self.__ontology_uri})
        self.__graph_store.query("CALL n10s.nsprefixes.add('time', $uri )", params={"uri": _TIME_ONTOLOGY_URI})
        self.__graph_store.query("CALL n10s.nsprefixes.add('mlsx', $uri)", params={"uri": _MLSX_ONTOLOGY_URI})

        # Load the ontologies
        self.__graph_store.query(
            "CALL n10s.onto.import.inline($ontology, 'Turtle')",
            params={"ontology": Path(self.__ontology_path).read_text()},
        )
        self.__graph_store.query("CALL n10s.onto.import.fetch($url, 'Turtle')", params={"url": _TIME_ONTOLOGY_URI})

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
        # Retrieves all classes in the log ontology,
        # along with any classes that have a first-degree subclass relationship with them.
        # It also retrieves any properties associated with each class, if they exist.
        nodes_with_props = self.__graph_store.query(
            """
            MATCH (c:n4sch__Class)
            WHERE c.uri STARTS WITH $log_ontology_uri
                OR EXISTS {
                    MATCH (c)<-[:n4sch__subClassOf]-(d:n4sch__Class)
                    WHERE d.uri STARTS WITH $log_ontology_uri
                }
            OPTIONAL MATCH (c)<-[:n4sch__domain]-(p:n4sch__Property)
            RETURN n10s.rdf.shortFormFromFullUri(c.uri) AS class,
                c.uri AS uri,
                apoc.map.fromPairs(
                    [pair IN COLLECT(
                        CASE WHEN p IS NOT NULL
                            THEN [n10s.rdf.shortFormFromFullUri(p.uri), p.n4sch__name]
                        ELSE NULL END
                    ) WHERE pair IS NOT NULL]
               ) AS properties
            """,
            params={"log_ontology_uri": self.__ontology_uri},
        )
        nodes_dict = {
            row["uri"]: Node(id=row["uri"], type=row["class"], properties=row["properties"]) for row in nodes_with_props
        }

        # Retrieves triples between the classes retrieved above,
        # representing ontological and structural relationships between them.
        triples = self.__graph_store.query(
            """
            MATCH (n:n4sch__Class)<-[:n4sch__domain]-(r:n4sch__Relationship)-[:n4sch__range]->(m:n4sch__Class)
            WHERE n.uri IN $node_uris AND m.uri IN $node_uris
            RETURN n.uri AS subject_uri,
                n10s.rdf.shortFormFromFullUri(r.uri) AS predicate,
                m.uri AS object_uri
            UNION
            MATCH (n:n4sch__Class)-[r]->(m:n4sch__Class)
            WHERE n.uri IN $node_uris AND m.uri IN $node_uris AND type(r) STARTS WITH 'n4sch'
            RETURN n.uri AS subject_uri, type(r) AS predicate, m.uri AS object_uri
            """,
            params={
                "node_uris": list(nodes_dict.keys()),
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

        result = GraphDocument(
            nodes=list(nodes_dict.values()),
            relationships=relationships,
            source=Document(
                page_content="Ontology graph",
                metadata={"ontology_uri": self.__ontology_uri},
            ),
        )
        return normalize_output_graph(result)

    @functools.cached_property
    def __ontology_uri(self) -> str:
        """Return the ontology URI from the ontology file."""
        return get_uri_from_ttl(self.__ontology_path)
