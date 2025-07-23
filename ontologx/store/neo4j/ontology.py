from pathlib import Path

from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph

from ontologx.config import Config
from ontologx.store import GraphDocument, Node, Relationship
from ontologx.store.neo4j.utils import normalize_input_graph, normalize_output_graph

_TIME_ONTOLOGY_URI = "http://www.w3.org/2006/time#"
_MLSCHEMA_ONTOLOGY_URI = "http://www.w3.org/ns/mls#"
_ONTOLOGY_PARAMS = {
    "subClassOfRel": "subClassOf",
    "subPropertyOfRel": "subPropertyOf",
    "domainRel": "domain",
    "rangeRel": "range",
}


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
        self.__graph_store.query("CALL n10s.graphconfig.init($params)", params={"params": _ONTOLOGY_PARAMS})
        self.__graph_store.query(
            f"""CREATE CONSTRAINT {self.__config.n10s_constraint_name} IF NOT EXISTS
            FOR (r:Resource) REQUIRE r.uri IS UNIQUE""",
        )

        # Set the namespaces
        self.__graph_store.query("CALL n10s.nsprefixes.add('olx', $uri)", params={"uri": self.__config.ontology_uri})
        self.__graph_store.query("CALL n10s.nsprefixes.add('time', $uri )", params={"uri": _TIME_ONTOLOGY_URI})
        self.__graph_store.query("CALL n10s.nsprefixes.add('mls', $uri)", params={"uri": _MLSCHEMA_ONTOLOGY_URI})

        # Load the ontologies
        self.__graph_store.query(
            "CALL n10s.onto.import.inline($ontology, 'Turtle')",
            params={"ontology": Path(self.__config.ontology_path).read_text()},
        )
        self.__graph_store.query("CALL n10s.onto.import.fetch($url, 'Turtle')", params={"url": _TIME_ONTOLOGY_URI})
        self.__graph_store.query("CALL n10s.onto.import.fetch($url, 'Turtle')", params={"url": _MLSCHEMA_ONTOLOGY_URI})

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
            params={"log_ontology_uri": self.__config.ontology_uri},
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
                metadata={"ontology_uri": self.__config.ontology_uri},
            ),
        )
        return normalize_output_graph(result)

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
        norm_graph = normalize_input_graph(graph)

        nodes_uris = [node.id for node in norm_graph.nodes]
        result = self.__graph_store.query(
            """
            MATCH (n)
            WHERE n.uri IN $uris
            WITH COLLECT(n) AS nodes
            CALL n10s.validation.shacl.validateSet(nodes)
            YIELD focusNode, nodeType, propertyShape, offendingValue, resultPath, severity
            WHERE resultPath <> 'uri' AND NOT n10s.rdf.shortFormFromFullUri(resultPath) STARTS WITH 'n4sch'
            RETURN focusNode, nodeType, propertyShape, offendingValue, resultPath, severity
            """,
            params={"uris": nodes_uris},
        )

        return len(result)
