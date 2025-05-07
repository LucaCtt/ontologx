FROM neo4j:2025.04.0

# Install APOC and n10s plugins
ENV NEO4J_HOME=/var/lib/neo4j
RUN mkdir -p $NEO4J_HOME/plugins && \
    cd $NEO4J_HOME/plugins && \
    wget https://github.com/neo4j/apoc/releases/download/2025.04.0/apoc-2025.04.0-core.jar && \
    wget https://github.com/neo4j-labs/neosemantics/releases/download/5.20.0/neosemantics-5.20.0.jar 

# Enable APOC and n10s in the configuration
ENV NEO4J_dbms_security_procedures_unrestricted="apoc.*,n10s.*"
ENV NEO4J_dbms_security_procedures_allowlist="apoc.*,n10s.*"
ENV NEO4J_AUTH=neo4j/password
ENV NEO4J_apoc_export_file_enabled=true
ENV NEO4J_apoc_import_file_enabled=true
ENV NEO4J_apoc_import_file_use__neo4j__config=true

# Enable triggers
ENV NEO4J_apoc_trigger_enabled=true

COPY ./neo4j-start.sh /neo4j-start.sh
RUN chmod +x /neo4j-start.sh

CMD ["./neo4j-start.sh"]