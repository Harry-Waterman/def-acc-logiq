# Harbour

Harbour serves as the centralized database for collecting data from the rest of the product.

## Overview

Harbour runs a Neo4j graph database instance using Docker Compose. This database acts as the central data collection point for the application.

## Database Configuration

The Neo4j database is configured with:

- **Image**: `neo4j:latest`
- **Container Name**: `neo4j`
- **Ports**:
  - `7474`: HTTP interface (Neo4j Browser)
  - `7687`: Bolt protocol (database connections)
- **Plugins**: APOC (Awesome Procedures on Cypher)
- **Volumes**:
  - `neo4j_data`: Persistent data storage
  - `neo4j_logs`: Log files
- **Restart Policy**: `unless-stopped`

## Environment Variables

The database credentials are loaded from the `.env` file in the project root:

- `NEO4J_USERNAME`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password

These are used to configure `NEO4J_AUTH` in the format `${NEO4J_USERNAME}/${NEO4J_PASSWORD}`.

## Setup

1. Ensure you have a `.env` file in the project root with the required credentials:
   ```
   NEO4J_USERNAME=your_username
   NEO4J_PASSWORD=your_password
   ```

2. Navigate to the Harbour directory:
   ```bash
   cd Harbour
   ```

3. Start the database:
   ```bash
   docker-compose up -d
   ```

4. Access the Neo4j Browser at `http://localhost:7474`

## Stopping the Database

To stop the database:
```bash
docker-compose down
```

To stop and remove volumes (⚠️ **WARNING**: This will delete all data):
```bash
docker-compose down -v
```

