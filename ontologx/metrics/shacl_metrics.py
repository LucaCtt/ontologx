"""SHACL metrics computation module."""

import functools

import pyshacl
import rdflib

from ontologx.store import GraphDocument


class SHACLMetrics:
    """Class to calculate SHACL metrics for a given graph."""

    def __init__(self, graph: GraphDocument, ontology_path: str, shacl_path: str) -> None:
        self.__data_graph = self.__convert_to_rdflib(graph)
        self.__ontology_graph = rdflib.Graph()
        self.__ontology_graph.parse(ontology_path, format="turtle")

        self.__shacl_graph = rdflib.Graph()
        self.__shacl_graph.parse(shacl_path, format="turtle")

    @functools.cached_property
    def violations_ratio(self) -> float:
        """Calculate the ratio of SHACL violations to total constraints."""
        total_violations = self.__total_violations
        total_constraints = self.__total_constraints

        if total_constraints == 0:
            return 0.0

        return total_violations / total_constraints

    @functools.cached_property
    def compliance_ratio(self) -> float:
        """Calculate the compliance ratio based on SHACL violations."""
        return 1.0 - self.violations_ratio

    @functools.cached_property
    def __total_violations(self) -> int:
        """Validate the graph against SHACL constraints.

        Returns:
            float: number of SHACL constraints that are violated

        """
        _, results_graph, _ = pyshacl.validate(
            self.__data_graph,
            shacl_graph=self.__shacl_graph,
            ont_graph=self.__ontology_graph,
            inference="rdfs",
        )
        if not isinstance(results_graph, rdflib.Graph):
            msg = "Validation results are not in rdflib.Graph format."
            raise TypeError(msg)

        return len(
            results_graph.query("""
            SELECT (COUNT(?violation) AS ?count)
            WHERE {
                ?violation a <http://www.w3.org/ns/shacl#ValidationResult> .
            }
            """),
        )

    @functools.cached_property
    def __total_constraints(self) -> int:
        """Get the total number of SHACL constraints."""
        query = """
        SELECT (COUNT(?shape) AS ?count)
        WHERE {
            ?shape a <http://www.w3.org/ns/shacl#NodeShape> .
        }
        """
        result = self.__shacl_graph.query(query)
        return len(result)

    def __convert_to_rdflib(self, graph: GraphDocument) -> rdflib.Graph:
        """Convert a GraphDocument to an rdflib Graph."""
        res = rdflib.Graph()

        for node in graph.nodes:
            node_uri = rdflib.URIRef(str(node.id))
            res.add((node_uri, rdflib.RDF.type, rdflib.URIRef(node.type)))

            for prop, value in node.properties.items():
                if prop == "uri" or prop.startswith("schema"):
                    continue

                res.add((node_uri, rdflib.URIRef(prop), rdflib.Literal(value)))

        for rel in graph.relationships:
            start_uri = rdflib.URIRef(str(rel.source.id))
            end_uri = rdflib.URIRef(str(rel.target.id))
            res.add((start_uri, rdflib.URIRef(rel.type), end_uri))

            for prop, value in rel.properties.items():
                if prop == "uri" or prop.startswith("schema"):
                    continue

                res.add((start_uri, rdflib.URIRef(prop), rdflib.Literal(value)))

        return res
