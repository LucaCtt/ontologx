"""Main entry point for the log knowledge graph builder.

Handles the instantiation of the parser and the backend,
the reading of the logs, and the construction of the knowledge graph.
"""

import logging

import boto3
import polars as pl
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_community.embeddings import InfinityEmbeddings
from rdflib import Graph
from rich.logging import RichHandler

from ontologx.agents.graph_connector import GraphConnectorContext, GraphConnectorInput, graph_connector_agent
from ontologx.settings import Settings
from ontologx.stores import GraphStore, VectorStore

# Setup logging
logging.basicConfig(
    format="%(message)s",
    handlers=[
        RichHandler(
            locals_max_string=200,
            tracebacks_code_width=200,
            tracebacks_width=None,
            omit_repeated_times=False,
            show_path=False,
        ),
    ],
)
logger = logging.getLogger("rich")
logger.setLevel(logging.DEBUG)
# Silence useless logs from Neo4j
logging.getLogger("neo4j").setLevel(logging.ERROR)

settings = Settings()

sts_client = boto3.client(
    "sts",
    config=Config(read_timeout=300),
    aws_access_key_id=settings.aws_access_key_id.get_secret_value() if settings.aws_access_key_id else None,
    aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() if settings.aws_secret_access_key else None,
)

# Assume the role
response = sts_client.assume_role(
    RoleArn=settings.aws_role_arn.get_secret_value() if settings.aws_role_arn else None,
    RoleSessionName="langchain-bedrock-session",
    DurationSeconds=60 * 60 * 1,  # 1 hour
)

# Extract the temporary credentials
credentials = response["Credentials"]

llm = ChatBedrockConverse(
    model=settings.llm_name,
    temperature=settings.llm_temperature,
    region_name=settings.aws_region,
    aws_access_key_id=credentials["AccessKeyId"],
    aws_secret_access_key=credentials["SecretAccessKey"],
    aws_session_token=credentials["SessionToken"],
)

embeddings = InfinityEmbeddings(model=settings.embeddings_name, infinity_api_url=settings.embeddings_url)

graph_store = GraphStore(url=settings.db_url)
vector_store = VectorStore(embeddings=embeddings)


def main() -> None:
    """Entry point for the log knowledge graph builder."""
    examples_graph = Graph()
    examples_graph.parse(settings.examples_path)
    graph_store.add_graph(examples_graph)

    ontology = Graph()
    ontology.parse(settings.ontology_path)

    events = pl.read_csv(settings.logs_path)
    # Normalize column names to lowercase and save into settings for later use
    events = events.rename({col: col.lower() for col in events.columns})

    """Run the log knowledge graph builder."""
    graph_connector_agent.invoke(
        input=GraphConnectorInput(events=events),
        context=GraphConnectorContext(llm=llm, ontology=ontology, vector_store=vector_store, graph_store=graph_store),
    )


if __name__ == "__main__":
    main()
