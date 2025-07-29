"""Module for managing event graphs in a Neo4j store."""

import uuid
from pathlib import Path

import neo4j
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector

from ontologx.config import Config
from ontologx.store import GraphDocument, Node, Relationship
from ontologx.store.neo4j.utils import normalize_input_graph, normalize_output_graph


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
            index_name="eventsIndex",
            node_label="olx__Event",
            embedding_node_property="n4sch__embedding",
            retrieval_query="""
            RETURN node.mls__implements AS text,
            score,
            {uri: node.uri, n4sch__runName: node.n4sch__runName, _embedding_: node.n4sch__embedding} AS metadata
            """,
        )

    def initialize(self) -> None:
        """Initialize the dataset by loading the examples and tests, and creating the vector index for events."""
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
            MATCH (d:mls__Dataset)-[:mls__hasPart]->(e:olx__Event)
            WHERE e.n4sch__embedding IS NULL AND d.uri STARTS WITH $examples_uri
            OPTIONAL MATCH (e)-[:olx__wasLoggedBy]->(s:olx__Source)
            SET e.n4sch__runName = ''
            RETURN elementId(e) AS id, e.mls__implements AS eventMessage, s.olx__sourceName AS sourceName,
            s.olx__sourceDevice AS sourceDevice
            """,
            params={
                "run_name": self.__config.run_name,
                "examples_uri": self.__config.examples_uri,
            },
        )

        texts = []
        for el in to_populate:
            context = {}
            if el.get("sourceName"):
                context["sourceName"] = el["sourceName"]
            if el.get("sourceDevice"):
                context["sourceDevice"] = el["sourceDevice"]

            texts.append(_compose_embeddings_text(el["eventMessage"], context))

        text_embeddings = self.__embeddings.embed_documents(texts)
        self.__graph_store.query(
            """
            UNWIND $data AS row
            MATCH (e:olx__Event)
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
            MATCH (d:mls__Dataset)-[:mls__hasPart]->(e:olx__Event)
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
        """Return a list of test documents from the dataset."""
        test_nodes = self.__graph_store.query(
            """
            MATCH (d:mls__Dataset)-[:mls__hasPart]->(e:olx__Event)
            WHERE d.uri STARTS WITH $tests_uri
            RETURN e.mls__implements as eventMessage, e.uri as uri
            ORDER BY e.uri
            """,
            params={"run_name": self.__config.run_name, "tests_uri": self.__config.tests_uri},
        )
        tests = []
        for test in test_nodes:
            graph = self.__get_subgraph_from_node(test["uri"])

            source_node = next((node for node in graph.nodes if node.type == "olx__Source"), None)
            context = {}
            if source_node:
                if source_node.properties.get("olx__sourceName"):
                    context["sourceName"] = source_node.properties["olx__sourceName"]
                if source_node.properties.get("olx__sourceDevice"):
                    context["sourceDevice"] = source_node.properties["olx__sourceDevice"]

            graph.source = Document(page_content=test["eventMessage"], metadata=context)
            tests.append(normalize_output_graph(graph))

        return tests

    def add_event_graph(self, graph: GraphDocument) -> None:
        """Add an event graph to the store.

        All the nodes will be tagged with the current run name,
        and for Event nodes the embedding will be added.

        Args:
            graph (GraphDocument): The event graph to add.

        """
        # Ensure all nodes have a unique ID and a URI property.
        # Do this before normalizing the graph, so the un-normalized graph
        # has consistent IDs and URIs if it is used somewhere else.
        for node in graph.nodes:
            node.id = f"{self.__config.out_uri}/{uuid.uuid4()}"
            node.properties["uri"] = node.id

        norm_graph = normalize_input_graph(graph)

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

        for node in norm_graph.nodes:
            node.properties["n4sch__runName"] = self.__config.run_name

            if node.type == "olx__Event":
                # This will raise an exception if the LLM produces an Event node without a message property.
                text = _compose_embeddings_text(
                    norm_graph.source.page_content,
                    norm_graph.source.metadata,
                )
                node.properties["n4sch__embedding"] = self.__embeddings.embed_query(text)

            self.__graph_store.query(
                f"""
                MATCH (d:mls__Dataset)
                WHERE d.uri STARTS WITH $out_dataset_uri
                CREATE (d)-[:mls__hasPart]->(n:{node.type} $props)
                """,
                params={"props": node.properties, "out_dataset_uri": self.__config.out_uri + "/out-dataset"},
            )

        for relationship in norm_graph.relationships:
            self.__graph_store.query(
                f"""
                MATCH (a {{uri: $source_uri}}), (b {{uri: $target_uri}})
                CREATE (a)-[:{relationship.type}]->(b)
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

        return [normalize_output_graph(self.__get_subgraph_from_node(doc.metadata["uri"])) for doc in relevant_docs]

    def __get_subgraph_from_node(self, node_uri: str, props_to_remove: list[str] | None = None) -> GraphDocument:
        """Get the subgraph of a node in the store.

        The subgraph will contain all the nodes and relationships connected to the given node, even indirectly.

        Args:
            node_uri (str): The URI of the node to get the subgraph from.
            props_to_remove (list[str] | None): The list of properties to remove from the nodes in the subgraph.
                If None, the default properties to remove are used.

        Returns:
            GraphDocument: The subgraph of the node, with nodes and relationships.

        Raises:
            ValueError: If no subgraph is found for the given node URI.

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
            type: HEAD([label IN LABELS(node) WHERE label <> 'Resource']),
            properties: apoc.map.removeKeys(PROPERTIES(node), $props_to_remove)
            }] AS nodes,
            [rel IN relationships | {
            source: STARTNODE(rel).uri,
            target: ENDNODE(rel).uri,
            type: TYPE(rel)
            }] AS relationships
            """,
            params={"node_uri": node_uri, "props_to_remove": props_to_remove},
        )

        if not nodes_subgraphs:
            msg = f"No subgraph found for node with URI: {node_uri}"
            raise ValueError(msg)

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
            source=Document(page_content=event_node.properties["mls__implements"], metadata=context),
        )
