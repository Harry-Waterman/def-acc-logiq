# Harbour API Server

API server for Harbour that receives JSON payloads from the Chrome extension and translates them to Neo4j database operations.

## Overview

This Express.js server acts as a bridge between the Chrome extension and the Neo4j graph database. It receives email data in JSON format and creates the corresponding graph structure in Neo4j.

## API Endpoints

### `POST /api/emails`

Receives email data from the Chrome extension and creates the graph structure in Neo4j.

**Request Body:**
```json
{
  "sender": "sender@example.com",
  "displayName": "Sender",
  "recipients": ["recipient@example.com"],
  "dateTime": "Thu 11/13/2025 4:46 PM",
  "attachments": ["file.txt"],
  "urls": ["https://example.com/link"],
  "flags": ["REQUESTING MONEY"],
  "score": 0.95,
  "installationId": "inst-001"
}
```

**Response:**
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

### `GET /health`

Health check endpoint to verify the API server is running.

**Response:**
```json
{
  "status": "ok",
  "service": "harbour-api"
}
```

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

2. Set environment variables in `.env` file (in project root):
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
PORT=3000
```

3. Start the server:
```bash
npm start
```

For development with auto-reload:
```bash
npm run dev
```

## Data Structure

The API automatically creates the following graph structure:

- **Email** nodes with `dateTime` (unique ID generated internally)
- **Address** nodes for FROM and TO email addresses
- **DisplayName** nodes for sender display names (with `name` property)
- **Domain** nodes extracted from email addresses and URLs
- **Url** nodes for URLs found in emails
- **Flag** nodes for email flags (PHISHING, SPAM, etc.) with `type` property
- **Score** nodes for risk scores with `value` property
- **installationId** and **userId** nodes

Relationships are created according to the schema:
- Email → FROM → Address
- Email → TO → Address
- Address → HAS_DISPLAY_NAME → DisplayName
- Address → HAS_DOMAIN → Domain
- Email → CONTAINS_URL → Url
- Url → HAS_DOMAIN → Domain
- Email → HAS_FLAG → Flag
- Email → HAS_Score → Score
- Email → OWNER → installationId
- userId → INSTALLED_BY → installationId

## Example Usage

### Using curl

```bash
curl -X POST http://localhost:3000/api/emails \
  -H "Content-Type: application/json" \
  -d '{
    "sender": "sender@example.com",
    "displayName": "Sender",
    "recipients": ["recipient@example.com"],
    "dateTime": "Thu 11/13/2025 4:46 PM",
    "attachments": ["file.txt"],
    "urls": ["https://example.com/link"],
    "flags": ["REQUESTING MONEY"],
    "score": 0.95,
    "installationId": "inst-001"
  }'
```

### Using JavaScript (Chrome Extension)

```javascript
const emailData = {
  sender: "sender@example.com",
  displayName: "Sender",
  recipients: ["recipient@example.com"],
  dateTime: "Thu 11/13/2025 4:46 PM",
  attachments: ["file.txt"],
  urls: ["https://example.com/link"],
  flags: ["REQUESTING MONEY"],
  score: 0.95,
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

