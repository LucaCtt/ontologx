#!/bin/bash

NEO4J_HOME=/var/lib/neo4j

mkdir -p $NEO4J_HOME/plugins
cd $NEO4J_HOME/plugins

# Download the plugins
wget https://github.com/neo4j/apoc/releases/download/2025.04.0/apoc-2025.04.0-core.jar
wget https://github.com/neo4j-labs/neosemantics/releases/download/5.20.0/neosemantics-5.20.0.jar 

# Start neo4j
neo4j