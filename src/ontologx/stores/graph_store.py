"""GraphDB store module for managing event graphs."""

from string import Template

import requests
from mitreattack.stix20.MitreAttackData import Tactic, Technique
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
            data={"query": template.safe_substitute(event=event)},
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
        tactics: list[Tactic] | None = None,
        techniques: list[Technique] | None = None,
    ) -> None:
        """Add a graph to the store.

        Args:
            event (str): The event associated with the graph.
            graph (Graph): The graph to add.
            tactics (list[Tactic], optional): The MITRE ATT&CK tactics associated with the event.
            techniques (list[Technique], optional): The MITRE ATT&CK techniques associated with the event.

        """
        namespaces = [f"PREFIX {prefix}: <{uri}>" for prefix, uri in graph.namespaces()]

        template = Template('<<$s $p $o>> mlsx:eventMessage "$event"^^xsd:string')
        if tactics:
            for tactic in tactics:
                template = Template(
                    template.template + f' ; mlsx:tactic "{tactic.name}"^^xsd:string',
                )
        if techniques:
            for technique in techniques:
                template = Template(
                    template.template + f' ; mlsx:technique "{technique.name}"^^xsd:string',
                )

        template = Template(template.template + " .")

        embedded_triples = [
            template.safe_substitute(
                s=s.n3(graph.namespace_manager),
                p=p.n3(graph.namespace_manager),
                o=o.n3(graph.namespace_manager),
                event=event,
                tactic=tactics[0].name if tactics else "",
                technique=techniques[0].name if techniques else "",
            )
            for s, p, o in graph.triples((None, None, None))
        ]

        template = Template("""\
            $namespaces
            PREFIX mlsx: <https://cyberseclab.unibs.it/mlsx/dict#>

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
