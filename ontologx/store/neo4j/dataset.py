import uuid
from pathlib import Path

import neo4j
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector

from ontologx.config import Config
from ontologx.store import GraphDocument, Node, Relationship


def _compose_embeddings_text(event: str, context: dict[str, str]) -> str:
    text = f"event: '{event.replace("'", "\\'")}'"

    for key, value in context.items():
        text += f", {key}: '{value.replace("'", "\\'")}'"

    return text


class Dataset:
    """The Dataset module is responsible for managing the event graphs in the store.

    Includes the loading of the examples and tests, and the search for similar events.
    """

    def __init__(self, config: Config, graph_store: Neo4jGraph, embeddings: Embeddings) -> None:
        self.__config = config
        self.__graph_store = graph_store
        self.__embeddings = embeddings

        self.__vector_index = Neo4jVector(
            embedding=self.__embeddings,
            username=self.__config.neo4j_username,
            password=self.__config.neo4j_password,
            url=self.__config.neo4j_url,
            index_name=self.__config.events_index_name,
            node_label="log__Event",
            embedding_node_property="n4sch__embedding",
            retrieval_query="""
            RETURN node.log__eventMessage AS text,
            score,
            {uri: node.uri, n4sch__runName: node.n4sch__runName, _embedding_: node.n4sch__embedding} AS metadata
            """,
        )

    def initialize(self) -> None:
        # Check if the examples are already loaded
        result = self.__graph_store.query(
            """
            MATCH (r:mls__Run {n4sch__runName: $run_name})-[:mls__hasInput]->(d:mls__Dataset)
            RETURN d
            LIMIT 1
            """,
            params={"run_name": self.__config.run_name},
        )
        if result:
            return

        # Load the examples
        # Note: n10s will not load the examples if they are already present in the store.
        # This is ok, as we want to load the examples again only when they change.
        self.__graph_store.query(
            "CALL n10s.rdf.import.inline($examples, 'Turtle')",
            params={"examples": Path(self.__config.examples_path).read_text()},
        )

        # Attach the examples to the current run
        self.__graph_store.query(
            """
            MATCH (d:mls__Dataset), (r:mls__Run {n4sch__runName: $run_name})
            WHERE d.uri STARTS WITH $examples_uri
            CREATE (r)-[:mls__hasInput]->(d)
            """,
            params={"run_name": self.__config.run_name, "examples_uri": self.__config.examples_uri},
        )

        # Create the index for the event messages
        self.__vector_index.create_new_index()

        # Populate the embeddings for the examples
        to_populate = self.__graph_store.query(
            """
            MATCH (d:mls__Dataset)-[:mls__hasPart]->(e:log__Event)
            WHERE e.n4sch__embedding IS NULL AND d.uri STARTS WITH $examples_uri
            OPTIONAL MATCH (e)-[:log__hasParameter]->(s:log__Source)
            SET e.n4sch__runName = ''
            RETURN elementId(e) AS id, e.log__eventMessage AS eventMessage, s.log__sourceName AS sourceName,
            s.log__sourceDevice AS sourceDevice
            """,
            params={
                "run_name": self.__config.run_name,
                "examples_uri": self.__config.examples_uri,
            },
        )
        texts = [
            _compose_embeddings_text(
                el["eventMessage"],
                {
                    "sourceName": el.get("sourceName", ""),
                    "sourceDevice": el["sourceDevice"] if el.get("sourceDevice") else "",  # Handle missing sourceDevice
                },
            )
            for el in to_populate
        ]
        text_embeddings = self.__embeddings.embed_documents(texts)
        self.__graph_store.query(
            """
            UNWIND $data AS row
            MATCH (e:log__Event)
            WHERE elementId(e) = row.id
            CALL db.create.setNodeVectorProperty(e, 'n4sch__embedding', row.embedding)
            """,
            params={
                "data": [
                    {"id": el["id"], "embedding": embedding}
                    for el, embedding in zip(to_populate, text_embeddings, strict=True)
                ],
            },
        )

        # Load the tests
        # Note: the test events should not have an embedding.
        self.__graph_store.query(
            "CALL n10s.rdf.import.inline($tests, 'Turtle')",
            params={"tests": Path(self.__config.tests_path).read_text()},
        )
        self.__graph_store.query(
            """
            MATCH (d:mls__Dataset), (r:mls__Run {n4sch__runName: $run_name})
            WHERE d.n4sch__runName IS NULL AND d.uri STARTS WITH $tests_uri
            MERGE (r)-[:mls__hasInput]->(d)
            """,
            params={
                "run_name": self.__config.run_name,
                "tests_uri": self.__config.tests_uri,
            },
        )
        self.__graph_store.query(
            """
            MATCH (d:mls__Dataset)-[:mls__hasPart]->(e:log__Event)
            WHERE e.n4sch__embedding IS NULL AND d.uri STARTS WITH $tests_uri
            SET e.n4sch__runName = ''
            SET e.n4sch__embedding = ''
            """,
            params={
                "run_name": self.__config.run_name,
                "tests_uri": self.__config.tests_uri,
            },
        )

    def tests(self) -> list[GraphDocument]:
        test_nodes = self.__graph_store.query(
            """
            MATCH (d:mls__Dataset)-[:mls__hasPart]->(e:log__Event)
            WHERE d.uri STARTS WITH $tests_uri
            RETURN e.log__eventMessage as message, e.uri as uri
            ORDER BY e.uri
            """,
            params={"run_name": self.__config.run_name, "tests_uri": self.__config.tests_uri},
        )
        tests = []
        for test in test_nodes:
            graph = self.__get_subgraph_from_node(test["uri"])

            source_node = next((node for node in graph.nodes if node.type == "Source"), None)
            context = (
                {
                    "sourceName": source_node.properties.get("log__sourceName", ""),
                    "sourceDevice": source_node.properties.get("log__sourceDevice", ""),
                }
                if source_node
                else {}
            )
            graph.source = Document(page_content=test["message"], metadata=context)
            tests.append(graph)

        return tests

    def add_event_graph(self, graph: GraphDocument) -> None:
        """Add an event graph to the store.

        All the nodes will be tagged with the current run name,
        and for Event nodes the embedding will be added.

        Args:
            graph (GraphDocument): The event graph to add.

        """
        # Check if result dataset exists, otherwise create it
        result_dataset = self.__graph_store.query(
            """
            MATCH (r:mls__Run {n4sch__runName: $run_name})-[:mls__hasOutput]->(d:mls__Dataset)
            RETURN d
            """,
            params={"run_name": self.__config.run_name},
        )
        if not result_dataset:
            self.__graph_store.query(
                """
                MATCH (r:mls__Run {n4sch__runName: $run_name})
                CREATE (d:mls__Dataset {uri: $out_dataset_uri})<-[:mls__hasOutput]-(r)
                """,
                params={"run_name": self.__config.run_name, "out_dataset_uri": self.__config.out_uri + "/out-dataset"},
            )

        for node in graph.nodes:
            node_id = f"{self.__config.out_uri}/{uuid.uuid4()}"
            node_type = node.type.replace(":", "__")
            node.id = node_id

            props = {k.replace(":", "__"): v for k, v in node.properties.items()} | {
                "uri": node.id,
                "n4sch__runName": self.__config.run_name,
            }

            if node_type == "log__Event":
                # This will raise an exception if the LLM produces an Event node without a message property.
                text = _compose_embeddings_text(
                    props["log__eventMessage"],
                    graph.source.metadata,
                )
                props["n4sch__embedding"] = self.__embeddings.embed_query(text)

            self.__graph_store.query(
                f"""
                MATCH (d:mls__Dataset)
                WHERE d.uri STARTS WITH $out_dataset_uri
                CREATE (d)-[:mls__hasPart]->(n:{node_type} $props)
                """,
                params={"props": props, "out_dataset_uri": self.__config.out_uri + "/out-dataset"},
            )

        for relationship in graph.relationships:
            self.__graph_store.query(
                f"""
                MATCH (a {{uri: $source_uri}}), (b {{uri: $target_uri}})
                CREATE (a)-[:{relationship.type.replace(":", "__")}]->(b)
                """,
                params={
                    "source_uri": relationship.source.id,
                    "target_uri": relationship.target.id,
                },
            )

    def events_mmr_search(
        self,
        event: str,
        context: dict | None = None,
        k: int = 3,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[GraphDocument]:
        """Search for similar events in the store.

        Args:
            event (str): The event message to search for.
            context (dict): The context to use for the search.
            k (int): The number of events to return.
            fetch_k (int): The number of events to pass to the MMR algorithm.
            lambda_mult (float): number between 0 and 1, that determines the trade-off between relevance and diversity.
                0 means maximum diversity, 1 means maximum relevance.

        Returns:
            list[GraphDocument]: The list of graphs of similar events,
                with the nodes they are connected to and their relationships.

        """
        # This filter is for retrieving examples only,
        # or examples + generated events.
        # Examples will have no run name but an embedding,
        # Tests will have neither. Generated events will have both.
        run_name_filter = (
            [{"n4sch__runName": {"$eq": ""}}]
            if not self.__config.generated_graphs_retrieval
            else [{"n4sch__runName": {"$eq": self.__config.run_name}}, {"n4sch__runName": {"$eq": ""}}]
        )

        query = _compose_embeddings_text(event, context or {})
        relevant_docs = self.__vector_index.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter={
                "$or": run_name_filter,
                "n4sch__embedding": {"$ne": ""},
            },
            # Examples will have no run name but an embedding.
            # Tests will have neither. Generated events will have both.
        )

        return [self.__get_subgraph_from_node(doc.metadata["uri"]) for doc in relevant_docs]

    def __get_subgraph_from_node(self, node_uri: str, props_to_remove: list[str] | None = None) -> GraphDocument:
        """Get the subgraph of a node in the store.

        The subgraph will contain all the nodes and relationships connected to the given node, even indirectly.
        """
        if props_to_remove is None:
            props_to_remove = []

        props_to_remove = [*props_to_remove, "n4sch__runName", "n4sch__embedding"]

        # Ugly but quite efficient. Also filters out the embedding property and the Resource label.
        nodes_subgraphs = self.__graph_store.query(
            """
            MATCH (n {uri: $node_uri})
            CALL apoc.path.subgraphAll(n, {labelFilter: '-mls__Dataset'})
            YIELD nodes, relationships
            RETURN
            [node IN nodes | {
            uri: node.uri,
            type: REPLACE(HEAD([label IN LABELS(node) WHERE label <> 'Resource']), '__', ':'),
            properties: apoc.map.fromPairs(
                [key IN KEYS(apoc.map.removeKeys(PROPERTIES(node), $props_to_remove)) |
                    [REPLACE(key, '__', ':'), apoc.map.removeKeys(PROPERTIES(node), $props_to_remove)[key]]
                ]
            )
            }] AS nodes,
            [rel IN relationships | {
            source: STARTNODE(rel).uri,
            target: ENDNODE(rel).uri,
            type: REPLACE(TYPE(rel), '__', ':')
            }] AS relationships
            """,
            params={"node_uri": node_uri, "props_to_remove": props_to_remove},
        )

        if not nodes_subgraphs:
            return GraphDocument(nodes=[], relationships=[], source=Document(page_content="", metadata={}))

        nodes_subgraph = nodes_subgraphs[0]

        # The neo4j date and time objects are quite problematic, as they are not JSON serializable.
        # This is a workaround to convert them to strings.
        for node in nodes_subgraph["nodes"]:
            for key, value in node["properties"].items():
                if isinstance(value, neo4j.time.DateTime | neo4j.time.Date | neo4j.time.Time):
                    node["properties"][key] = value.iso_format()

        def get_node_id(uri: str) -> str:
            """Get the node id from the node uri."""
            final_str = uri.split("/")[-1]

            if "#" in final_str:
                return final_str.split("#")[-1]

            return final_str

        nodes_dict = {
            node["uri"]: Node(id=get_node_id(node["uri"]), type=node["type"], properties=node["properties"])
            for node in nodes_subgraph["nodes"]
        }

        relationships = (
            [
                Relationship(
                    source=nodes_dict[relationship["source"]],
                    target=nodes_dict[relationship["target"]],
                    type=relationship["type"],
                )
                for relationship in nodes_subgraph["relationships"]
            ]
            if "relationships" in nodes_subgraph
            else []  # The node may not have any relationships
        )

        # Create the context from the event source, if present.
        event_node = next(node for node in nodes_dict.values() if node.type == "olx__Event")
        source_node = next((node for node in nodes_dict.values() if node.type == "olx__Source"), None)
        context = {key: value for key, value in source_node.properties.items() if key != "uri"} if source_node else {}

        return GraphDocument(
            nodes=list(nodes_dict.values()),
            relationships=relationships,
            source=Document(page_content=event_node.properties["olx__eventMessage"], metadata=context),
        )
