FROM neo4j:2025.04.0

# Install APOC and n10s plugins
ENV NEO4J_HOME=/var/lib/neo4j
RUN mkdir -p $NEO4J_HOME/plugins && \
    cd $NEO4J_HOME/plugins && \
    wget https://github.com/neo4j/apoc/releases/download/2025.04.0/apoc-2025.04.0-core.jar -O apoc.jar && \
    wget https://github.com/neo4j-labs/neosemantics/releases/download/5.20.0/neosemantics-5.20.0.jar -O n10s.jar

# Enable APOC and n10s in the configuration
ENV NEO4J_dbms_security_procedures_unrestricted "apoc.*,n10s.*"
ENV NEO4J_dbms_security_procedures_allowlist "apoc.*,n10s.*"
ENV NEO4J_AUTH=neo4j/password

# Enable triggers
ENV NEO4J_apoc_trigger_enabled=true