import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { createEmailGraph } from './services/neo4jService.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'harbour-api' });
});

// Main endpoint to receive email data from Chrome extension
app.post('/api/emails', async (req, res) => {
  try {
    const emailData = req.body;
    
    // Validate required fields
    if (!emailData) {
      return res.status(400).json({ error: 'Request body is required' });
    }

    // Process and create graph in Neo4j
    const result = await createEmailGraph(emailData);
    
    res.status(201).json({
      success: true,
      message: 'Email data processed and stored in Neo4j',
      data: result
    });
  } catch (error) {
    console.error('Error processing email data:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Internal server error'
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Harbour API server running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});

