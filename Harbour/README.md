# Harbour

Harbour serves as the centralized database for collecting data from the rest of the product.

## Overview

Harbour runs a Neo4j graph database instance and an API server using Docker Compose. The database acts as the central data collection point for the application, and the API server translates JSON payloads from the Chrome extension into Neo4j graph operations.

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

## Database Schema

The following diagram illustrates the Neo4j graph database schema structure:

```mermaid
flowchart LR
 subgraph Database["Database"]
        Email("Email
        -----
        id: id
        dateTime: dateTime")
        Address("Address
        -----
        email: string")
        DisplayName("DisplayName
        -----
        name: string")
        Domain("Domain
        -----
        name: string")
        Url("Url
        -----
        id: id")
        Flag("Flag
        -----
        type: string")
        Score("Score
        -----
        value: int")
        installationId("installationId
        -----
        id: id")
  end

    Email -- FROM --> Address
    Email -- TO --> Address
    Address -- HAS_DISPLAY_NAME --> DisplayName
    Address -- HAS_DOMAIN --> Domain
    Email -- CONTAINS_URL --> Url
    Url -- HAS_DOMAIN --> Domain
    Email -- HAS_FLAG --> Flag
    Email -- HAS_SCORE --> Score
    Email -- OWNER --> installationId
```

## Environment Variables

All environment variables are loaded from the `.env` file in the project root. The following variables are used by Harbour services:

### Variables

- `NEO4J_USERNAME`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
   - These are used to configure `NEO4J_AUTH` in the format `${NEO4J_USERNAME}/${NEO4J_PASSWORD}`.

- `NEO4J_URI`: Neo4j connection URI
  - Used by API Server and NeoDash for database connections
  
- `NEO4J_DB`: Neo4j database name
  - Used by NeoDash and dashboard initialization
  - Specifies which database to connect to and store dashboards in
  
- `NEO4J_HOST`: Neo4j host address for NeoDash standalone mode (default: `neo4j`)
  - Used by NeoDash when running in standalone mode
  
- `NEO4J_PORT`: Neo4j port for NeoDash standalone mode (default: `7687`)
  - Used by NeoDash when running in standalone mode
  - Specifies the Bolt protocol port

## Services

Harbour includes four services:

1. **Neo4j Database**: Graph database for storing email data
2. **NeoDash Dashboard**: Visualization dashboard for Neo4j
3. **Dashboard Initialization**: One-time service that seeds the pre-configured dashboard on startup
4. **API Server**: REST API for receiving data from Chrome extension

## Architecture & Component Communication

Harbour is built as a microservices architecture using Docker Compose, with all services connected via a shared Docker network. The following diagram illustrates how components are connected and communicate:

```mermaid
flowchart TB
    subgraph External["External Components"]
        Chrome["Chrome Extension"]
        User["User Browser"]
    end
    
    subgraph Harbour["Harbour Docker Network (harbour-network)"]
        API["API Server<br/>(harbour-api)"]
        Neo4j["Neo4j Database<br/>(neo4j)"]
        NeoDash["NeoDash Dashboard<br/>(neodash)"]
        DashboardInit["Dashboard Init<br/>(dashboard-init)"]
    end
    
    Chrome -->|HTTP POST<br/>JSON Payload<br/>Port: 3000| API
    User -->|HTTP<br/>Browser UI<br/>Ports: 7474| Neo4j
    User -->|HTTP<br/>Dashboard UI<br/>Port: 5005| NeoDash
    
    API -->|Bolt Protocol<br/>Cypher Queries<br/>Ports: 7687| Neo4j
    NeoDash -->|Bolt Protocol<br/>Read Queries<br/>Ports: 7687| Neo4j
    DashboardInit -->|Bolt Protocol<br/>Seed Dashboard<br/>Ports: 7687| Neo4j
    
    style Harbour fill:#e1f5ff
    style API fill:#fff4e1
    style Neo4j fill:#e8f5e9
    style NeoDash fill:#f3e5f5
```

### Network Architecture

All Harbour services run within a Docker bridge network called `harbour-network`. This network enables:

- **Service Discovery**: Services can communicate using container names (e.g., `neo4j:7687`)
- **Isolation**: Services are isolated from other Docker networks
- **Internal Communication**: Services communicate internally without exposing ports externally (except where needed)

### Component Communication Flow

#### 1. Chrome Extension → API Server

**Protocol**: HTTP/REST  
**Port**: `3000` (exposed to host)  
**Endpoint**: `POST /api/emails`

The Chrome extension sends email data as JSON payloads to the API server:

```http
POST http://localhost:3000/api/emails
Content-Type: application/json

{
  "classification": {
    "score": "87",
    "flags": ["Suspicious urls", "requests for personal information"]
  },
  "email": {
    "sender": {
      "displayName": "Sender",
      "email": "sender@example.com"
    },
    "recipients": ["recipient@example.com"],
    "sentTime": "Thu 11/13/2025 4:46 PM",
    "attachments": ["file.txt"],
    "urls": ["https://example.com/link"]
  },
  "installationId": "inst-001"
}
```

**Communication Details**:
- The API server uses Express.js with CORS enabled to accept cross-origin requests
- Requests are validated and processed asynchronously
- The API server responds with JSON containing the created email ID and status

#### 2. API Server → Neo4j Database

**Protocol**: Bolt (Neo4j's binary protocol)  
**Internal URI**: `bolt://neo4j:7687` (uses container name for service discovery)  
**External URI**: `bolt://localhost:7687` (for external clients)

The API server uses the `neo4j-driver` library to communicate with Neo4j:

**Connection Flow**:
1. API server creates a Neo4j driver instance using credentials from environment variables
2. For each email submission, the API server:
   - Opens a session with Neo4j
   - Executes Cypher queries to create/merge nodes and relationships
   - Closes the session after processing

**Key Operations**:
- `MERGE` operations to create nodes if they don't exist (idempotent)
- Relationship creation between nodes
- Transaction management for data consistency

**Service Discovery**: The API server connects to Neo4j using the container name `neo4j` instead of `localhost`, allowing Docker to resolve the service within the network.

#### 3. NeoDash → Neo4j Database

**Protocol**: Bolt  
**Connection**: `bolt://localhost:7687` (when accessed from host) or `bolt://neo4j:7687` (internal)

NeoDash connects to Neo4j to:
- Execute read-only Cypher queries for visualization
- Build interactive dashboards
- Display graph data in various chart formats

**Configuration**: NeoDash runs in standalone mode and is configured via environment variables:
- `NEO4J_URI`: Connection string 
- `NEO4J_USER`: Username 
- `NEO4J_PASSWORD`: Password
- `NEO4J_DATABASE`: Database name
- `standalone`: Set to `true` to enable standalone mode
- `standaloneProtocol`: Protocol for standalone mode
- `standaloneHost`: Host for standalone mode
- `standalonePort`: Port for standalone mode 
- `standaloneDashboardName`: Name of the pre-seeded dashboard
- `standaloneDatabase`: Database for standalone dashboard

**Standalone Mode**: When enabled, NeoDash automatically connects to Neo4j and loads the pre-configured "Harbour Dashboard" on startup, eliminating the need for manual connection configuration.

#### 4. User Browser → Neo4j Browser

**Protocol**: HTTP  
**Port**: `7474` (exposed to host)  
**URL**: `http://localhost:7474`

Users can directly access Neo4j Browser to:
- Execute Cypher queries
- Visualize graph data
- Manage database schema

### Service Dependencies

The services have the following startup dependencies:

```
Neo4j Database (no dependencies)
    ↓
Dashboard Init (depends_on: neo4j [healthy])
    ↓
NeoDash (depends_on: neo4j, dashboard-init)
API Server (depends_on: neo4j)
```

**Dependency Behavior**:
- Docker Compose ensures Neo4j starts before dependent services
- The `dashboard-init` service waits for Neo4j to be healthy before seeding the dashboard
- NeoDash waits for both Neo4j and `dashboard-init` to complete, ensuring the dashboard is ready on first access
- Services should implement retry logic for database connections (currently the API server connects immediately on startup)
- If Neo4j is not ready, dependent services may fail to connect initially

### Data Flow: Email Submission

The complete data flow when a Chrome extension submits email data:

1. **Chrome Extension** → Sends HTTP POST request with JSON payload to `http://localhost:3000/api/emails`
2. **API Server** (`server.js`) → Receives request, validates payload
3. **API Server** (`neo4jService.js`) → Processes email data:
   - Generates unique email ID
   - Extracts domains from email addresses and URLs
   - Creates/merges nodes (Email, Address, Domain, Url, Flag, Score, etc.)
   - Creates relationships between nodes
4. **Neo4j Database** → Stores graph structure persistently
5. **API Server** → Returns success response with email ID
6. **Chrome Extension** → Receives confirmation

### Port Mapping

| Service | Internal Port | External Port | Protocol | Purpose |
|---------|--------------|---------------|----------|---------|
| Neo4j | 7474 | 7474 | HTTP | Neo4j Browser UI |
| Neo4j | 7687 | 7687 | Bolt | Database connections |
| API Server | 3000 | 3000 | HTTP | REST API endpoints |
| NeoDash | 5005 | 5005 | HTTP | Dashboard UI |

**Note**: Ports are mapped to the host machine, allowing external access. Internal communication between containers uses container names and internal ports.

### Environment Variable Sharing

All services share environment variables from the `.env` file in the project root:

- **Neo4j Service**: Uses `NEO4J_USERNAME` and `NEO4J_PASSWORD` to configure authentication
- **API Server**: Uses `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` for database connections
- **NeoDash**: Uses `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DB`, `NEO4J_HOST`, and `NEO4J_PORT` for connection and standalone mode configuration
- **Dashboard Init**: Uses `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `NEO4J_DB` for seeding the dashboard

This ensures consistent authentication and configuration across all services.

## Accessing the UIs

Once the Docker stack is running, you can access the following interfaces:

### Neo4j Browser
- **URL**: `http://localhost:7474`
- **Purpose**: Interactive Cypher query interface and graph visualization
- **Login**: Use your Neo4j credentials from the `.env` file

### NeoDash Dashboard
- **URL**: `http://localhost:5005`
- **Purpose**: Interactive visualization dashboard for Harbour email data
- **Connection Settings**:
  - **URI**: `bolt://localhost:7687`
  - **Username/Password**: Use your Neo4j credentials from the `.env` file
- **Pre-seeded Dashboard**: The "Harbour Dashboard" is automatically loaded on first startup

#### Dashboard Overview

The Harbour Dashboard provides a comprehensive view of email data with the following visualizations:

1. **Graph Visualization**
   - Interactive graph showing the complete email network structure
   - Displays nodes for: Email, Address, DisplayName, Domain, Url, Flag, Score, and installationId
   - Shows relationships: FROM, TO, HAS_DISPLAY_NAME, HAS_DOMAIN, CONTAINS_URL, HAS_FLAG, HAS_SCORE, OWNER
   - Auto-refreshes every 30 seconds
   - Fullscreen mode enabled

2. **Emails Table**
   - Lists all emails with their risk scores and associated flags
   - Columns: Email ID, Score, Flags
   - Sorted by score (highest risk first)

3. **Flags Distribution (Pie Chart)**
   - Visual breakdown of email flags (e.g., "REQUESTING MONEY", etc.)
   - Shows count of emails per flag type
   - Sorted by value for easy identification of common threats

4. **Email Classification (Bar Chart)**
   - Risk classification breakdown:
     - **Benign**: Score 0-50
     - **Suspicious**: Score 51-80
     - **Phishing**: Score 81-100
   - Count of emails in each risk category

The dashboard is automatically seeded when you first start the Docker stack via the `dashboard-init` service, which loads the dashboard configuration from `dashboards/habour_overview.json`.

#### Dashboard Files

- **Dashboard Configuration**: `dashboards/habour_overview.json`
  - Contains the complete dashboard definition including pages, reports, queries, and visualizations
  - Can be edited to customize the dashboard layout and queries

- **Seeding Script**: `scripts/seed_dashboard.py`
  - Python script that loads the dashboard JSON into Neo4j
  - Automatically runs on startup via the `dashboard-init` Docker service
  - Waits for Neo4j to be healthy before seeding

To customize the dashboard:
1. Edit `dashboards/habour_overview.json` with your desired changes
2. Restart the Docker stack: `docker-compose restart dashboard-init`
3. Or manually run the seed script: `docker-compose run --rm dashboard-init`

### Harbour API Server
- **URL**: `http://localhost:3000`
- **Purpose**: REST API endpoint for receiving email data from Chrome extension
- **Endpoints**:
  - `POST /api/emails`: Receive and process email data
  - `GET /health`: Health check endpoint
- **Documentation**: See `api/README.md` for detailed API documentation

## Setup

1. Ensure you have a `.env` file in the project root with the required credentials. This can be done by copying `.env.example` which is prefilled with development defaults.

2. Navigate to the Harbour directory:
   ```bash
   cd Harbour
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Wait for initialisation:
   - The `dashboard-init` service will automatically seed the "Harbour Dashboard" on first startup
   - This is a one-time operation that runs after Neo4j becomes healthy
   - You can monitor the progress with: `docker logs dashboard-init`

5. Access the services:
   - Neo4j Browser: `http://localhost:7474`
   - NeoDash Dashboard: `http://localhost:5005` (pre-configured dashboard will be loaded automatically)
   - Harbour API: `http://localhost:3000`

## Loading Test Data

To populate the database with sample test data that matches the schema:

1. **Using Neo4j Browser** (Recommended):
   - Open Neo4j Browser at `http://localhost:7474`
   - Copy the contents of `test-data.cypher`
   - Paste into the query editor and execute

2. **Using cypher-shell** (Command Line):
   ```bash
   docker exec -i neo4j cypher-shell -u ${NEO4J_USERNAME} -p ${NEO4J_PASSWORD} < test-data.cypher
   ```

The test data includes:
- 5 sample emails with various risk levels
- Multiple addresses, domains, and URLs
- Flags and scores for phishing detection
- User and installation relationships

## Stopping the Service

To stop the database:
```bash
docker-compose down
```

To stop and remove volumes (⚠️ **WARNING**: This will delete all data):
```bash
docker-compose down -v
```

