"""GraphDB store module for managing event graphs."""

from string import Template

import requests
from rdflib import Graph

_REPO_TTL = """\
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>.
@prefix rep: <http://www.openrdf.org/config/repository#>.
@prefix sail: <http://www.openrdf.org/config/sail#>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

<#ontologx> a rep:Repository;
  rep:repositoryID "ontologx";
  rep:repositoryImpl [
    rep:repositoryType "graphdb:SailRepository";
    <http://www.openrdf.org/config/repository/sail#sailImpl> [
      <http://www.ontotext.com/config/graphdb#disable-sameAs> "false";
      sail:sailType "graphdb:Sail";
    ];
  ];
  rdfs:label "ontologx"^^xsd:string.
"""


class GraphStore:
    """Abstract base class for a store module that manages the storage and retrieval of event graphs."""

    def __init__(self, url: str) -> None:
        # Create repository
        requests.post(
            url + "/rest/repositories",
            files={"config": ("repo.ttl", _REPO_TTL, "text/turtle")},
            timeout=10,
        )
        self.__url = url

    def get_graph(self, event: str) -> Graph:
        """Retrieve a graph associated with a specific event.

        Args:
            event (str): The event to retrieve the graph for.

        Returns:
            Graph: The retrieved graph.

        """
        template = Template("""
            PREFIX mlsx: <https://cyberseclab.unibs.it/mlsx/dict#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            CONSTRUCT {?s ?p ?o} WHERE {
                <<?s ?p ?o>> mlsx:eventMessage "$event"^^xsd:string .
            }
        """)

        res = requests.post(
            self.__url + "/repositories/ontologx",
            data={"query": template.safe_substitute(event=event.replace('"', "").replace("'", "").replace("\n", " "))},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "text/turtle",
            },
            timeout=10,
        )
        graph = Graph()
        graph.parse(data=res.text, format="turtle")

        return graph

    def add_graph(
        self,
        event: str,
        graph: Graph,
        metadata: dict | None = None,
    ) -> None:
        """Add a graph to the store.

        Args:
            event (str): The event associated with the graph.
            graph (Graph): The graph to add.
            metadata (dict, optional): Additional metadata associated with the graph.

        """
        namespaces = [f"PREFIX {prefix}: <{uri}>" for prefix, uri in graph.namespaces() if prefix != ""]

        template = Template('<<$s $p $o>> mlsx:eventMessage "$event"^^xsd:string')
        for key, value in (metadata or {}).items():
            key_norm = key.replace("_", " ").title().replace(" ", "")

            if isinstance(value, list):
                for v in value:
                    template = Template(
                        template.template + f' ; mlsx:{key_norm} "{v}"^^xsd:string',
                    )
            else:
                template = Template(
                    template.template + f' ; mlsx:{key_norm} "{value}"^^xsd:string',
                )

        template = Template(template.template + " .")

        embedded_triples = [
            template.safe_substitute(
                s=s.n3(graph.namespace_manager),
                p=p.n3(graph.namespace_manager),
                o=o.n3(graph.namespace_manager),
                event=event.replace('"', "").replace("'", "").replace("\n", " "),
            )
            for s, p, o in graph.triples((None, None, None))
        ]

        template = Template("""\
            $namespaces
            PREFIX mlsx: <https://cyberseclab.unibs.it/mlsx/dict#>
            PREFIX : <http://cyberseclab.unibs.it/ontologx/run#>

            INSERT DATA { $triples }\
        """)

        requests.post(
            self.__url + "/repositories/ontologx/statements",
            data=template.safe_substitute(
                triples="\n".join(embedded_triples),
                namespaces="\n".join(namespaces),
            ),
            headers={"Content-Type": "application/sparql-update"},
            timeout=10,
        )
