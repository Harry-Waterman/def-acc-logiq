# def-acc-logiq

Team Logiq's submission to the def/acc hackathon - A comprehensive phishing email detection system using Small Language Models (SLMs) with local inference and graph-based analytics.

## Overview

This project provides an end-to-end solution for detecting phishing emails using browser-based Small Language Models (SLMs). The system consists of:

- **Chrome Extension**: Local, privacy-preserving phishing detection using WebLLM
- **Harbour**: Neo4j graph database backend for data collection and analytics
- **Evaluation Tools**: Model accuracy and repeatability benchmarking
- **Benchmark Analysis**: Comprehensive performance analysis and visualization

## Architecture

```
┌─────────────────┐
│ Chrome Extension│  (Local SLM Inference)
│  (WebLLM)       │
└────────┬────────┘
         │ HTTP POST
         ▼
┌─────────────────┐
│  Harbour API    │  (Express.js)
└────────┬────────┘
         │ Cypher Queries
         ▼
┌─────────────────┐
│  Neo4j Database │  (Graph Storage)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  NeoDash        │  (Visualization)
└─────────────────┘
```

## Quick Start

### Prerequisites

- **Node.js** (v16+)
- **Docker** and **Docker Compose** for Harbour backend
- **Chrome Browser** for the extension

### 1. Chrome Extension Setup

The extension uses WebLLM to run SLM models locally in the browser for privacy-preserving phishing detection.

```bash
npm install
npm run build
```

This will:
- Install dependencies
- Build the extension

See [chrome-extension/README.md](chrome-extension/README.md) for detailed setup instructions.

### 2. Harbour Backend Setup

Harbour provides the graph database backend for storing and analyzing email data.

```bash
cd Harbour
# Ensure .env file exists (copy from .env.example if needed)
docker-compose up -d
```

Access the services:
- **Neo4j Browser**: http://localhost:7474
- **NeoDash Dashboard**: http://localhost:5005
- **API Server**: http://localhost:3000

See [Harbour/README.md](Harbour/README.md) for detailed documentation.

### 3. Load Extension in Chrome

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `chrome-extension/dist/` directory

## Components

### Chrome Extension

Browser-based phishing email detection using WebLLM. Runs models locally without sending data to external servers.

**Features:**
- Local model inference
- Real-time email analysis
- Risk scoring and flagging
- Integration with Harbour backend

**Documentation**: [chrome-extension/README.md](chrome-extension/README.md)

### Harbour

Centralized graph database system for collecting and analyzing email data.

**Components:**
- **Neo4j Database**: Graph database for email relationships
- **API Server**: REST API for receiving data from Chrome extension
- **NeoDash Dashboard**: Interactive visualization dashboard
- **Dashboard Initialization**: Automated dashboard seeding

**Documentation**: [Harbour/README.md](Harbour/README.md)  
**API Documentation**: [Harbour/api/README.md](Harbour/api/README.md)

### Evaluation Module

Tools for assessing SLM model performance on phishing email detection.

**Features:**
- Accuracy evaluation against ground truth labels
- Repeatability benchmarking for consistency assessment
- Dataset integration support

**Documentation**: [evaluation/README.md](evaluation/README.md)

### Benchmark Analysis

Comprehensive analysis and visualization of model benchmark results.

**Features:**
- Comparative performance visualizations
- Metrics extraction and analysis
- Repeatability analysis
- Model selection recommendations

**Documentation**: [benchmark-analysis/README.md](benchmark-analysis/README.md)

## Project Structure

```
def-acc-logiq/
├── chrome-extension/      # Chrome extension with WebLLM integration
│   ├── src/               # Source code
│   ├── dist/              # Built extension (load this in Chrome)
│   └── README.md          # Extension setup guide
│
├── Harbour/               # Graph database backend
│   ├── api/               # Express.js API server
│   ├── dashboards/        # NeoDash dashboard configurations
│   ├── scripts/           # Database seeding scripts
│   └── README.md          # Harbour documentation
│
├── evaluation/            # Model evaluation tools
│   ├── config.py          # Evaluation configuration
│   ├── test_with_dataset.py
│   └── README.md          # Evaluation documentation
│
├── benchmark-analysis/    # Benchmark analysis tools
│   ├── analyze_benchmarks.py
│   ├── visualizations/    # Generated charts and graphs
│   └── README.md          # Analysis documentation
│
├── benchmark-results/     # Benchmark JSON results
├── Report/                # Project reports and diagrams
├── prompts/               # System prompts for models
└── requirements.txt       # Python dependencies
```
