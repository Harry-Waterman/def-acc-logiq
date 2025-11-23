# Harbour API Server

API server for Harbour that receives JSON payloads from the Chrome extension and translates them to Neo4j database operations.

## Overview

This Express.js server acts as a bridge between the Chrome extension and the Neo4j graph database. It receives email data in JSON format and creates the corresponding graph structure in Neo4j.

The server includes:
- **CORS support**: Enabled to accept cross-origin requests from the Chrome extension
- **JSON parsing**: Automatic parsing of JSON request bodies
- **Error handling**: Comprehensive error handling with descriptive error messages
- **Health checks**: Built-in health check endpoint for monitoring

## API Endpoints

### `POST /api/emails`

Receives email data from the Chrome extension and creates the graph structure in Neo4j.

**Request Body:**
```json
{
  classification: 
  {
    score: "87",
    flags: 
      [ "Suspicious urls",
      "requests for personal information"
      ]
  },
  email: 
  {
    sender: 
    {
      displayName: "geeksoutfit",
      email: "support@geeksoutfit.com"
    },
    "recipients": ["recipient@example.com"],
    "sentTime": "Thu 11/13/2025 4:46 PM",
    "urls": ["https://example.com/link"],
    "attachments": ["file.txt"]
  },
  "installationId": "inst-001"
}
```

**Response (Success - 201):**
```json
{
  "success": true,
  "message": "Email data processed and stored in Neo4j",
  "data": {
    "emailId": "a1b2c3d4e5f6g7h8",
    "status": "created",
    "timestamp": "2024-01-15T10:30:00.000Z"
  }
}
```

**Response (Error - 400):**
```json
{
  "error": "Request body is required"
}
```

**Response (Error - 500):**
```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

### `GET /health`

Health check endpoint to verify the API server is running.

**Response (200):**
```json
{
  "status": "ok",
  "service": "harbour-api"
}
```

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Returned when the request body is missing or invalid
- **500 Internal Server Error**: Returned when there's an error processing the email data or connecting to Neo4j

All error responses include a descriptive error message in the response body. The API validates:
- Request body presence
- Email object structure
- Sender information (required)
- Required fields for graph creation

If validation fails or Neo4j operations fail, appropriate error responses are returned with details about what went wrong.

## Environment Variables

The API server uses the following environment variables (loaded from `.env` file in project root):

- `NEO4J_URI`: Neo4j connection URI (default: `bolt://neo4j:7687`)
- `NEO4J_USERNAME`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `PORT`: API server port (default: `3000`)

## Running the Server

### Using Docker Compose (Recommended)

The API server is included in the `docker-compose.yml` file. Start all services:

```bash
cd Harbour
docker-compose up -d
```

The API will be available at `http://localhost:3000`

### Running Locally

1. Install dependencies:
```bash
cd Harbour/api
npm install
```

2. Set environment variables in `.env` file (in project root). You can copy `.env.example` which is prefilled with development defaults

3. Start the server:
```bash
npm start
```

For development with auto-reload:
```bash
npm run dev
```

## Data Structure

The API automatically creates the following graph structure in Neo4j:

### Nodes

- **Email** nodes with:
  - `id`: Unique identifier (generated from sender, recipients, and sentTime)
  - `dateTime`: Timestamp when the email was sent
  
- **Address** nodes for FROM and TO email addresses:
  - `id`: The email address (e.g., "sender@example.com")
  
- **DisplayName** nodes for sender display names:
  - `name`: The display name (e.g., "John Doe")
  
- **Domain** nodes extracted from email addresses and URLs:
  - `id`: The domain name (e.g., "example.com")
  
- **Url** nodes for URLs found in emails:
  - `id`: The full URL
  
- **Flag** nodes for email flags:
  - `type`: The flag type (e.g., "Suspicious urls", "requests for personal information")
  
- **Score** nodes for risk scores:
  - `value`: The numeric risk score (as string)
  
- **installationId** nodes:
  - `id`: The installation identifier

### Relationships

- `Email` → `FROM` → `Address` (sender email address)
- `Email` → `TO` → `Address` (recipient email addresses)
- `Address` → `HAS_DISPLAY_NAME` → `DisplayName` (sender display name)
- `Address` → `HAS_DOMAIN` → `Domain` (domain from email addresses)
- `Email` → `CONTAINS_URL` → `Url` (URLs found in email)
- `Url` → `HAS_DOMAIN` → `Domain` (domain from URLs)
- `Email` → `HAS_FLAG` → `Flag` (email flags)
- `Email` → `HAS_SCORE` → `Score` (risk score)
- `Email` → `OWNER` → `installationId` (installation that owns the email)

**Note**: Future feature - `userId` nodes and `installationId → INSTALLED_BY → userId` relationships will be added to link multiple installations to the same user.

All nodes use `MERGE` operations, ensuring idempotency - duplicate submissions won't create duplicate nodes.

## Example Usage

### Using curl

```bash
curl -X POST http://localhost:3000/api/emails \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Using JavaScript (Chrome Extension)

```javascript
const emailData = {
  classification: {
    score: "87",
    flags: ["Suspicious urls", "requests for personal information"]
  },
  email: {
    sender: {
      displayName: "Sender",
      email: "sender@example.com"
    },
    recipients: ["recipient@example.com"],
    sentTime: "Thu 11/13/2025 4:46 PM",
    attachments: ["file.txt"],
    urls: ["https://example.com/link"]
  },
  installationId: "inst-001"
};

fetch('http://localhost:3000/api/emails', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(emailData)
})
.then(response => response.json())
.then(data => console.log('Success:', data))
.catch(error => console.error('Error:', error));
```

