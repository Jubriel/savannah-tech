# Optimized Water Treatment Monitoring System

A robust, scalable system for monitoring water treatment sensor data, detecting anomalies, and generating AI-powered summaries.

## 🚀 Key Features

- **Real-time Monitoring**: Continuous sensor data simulation and processing
- **Advanced Anomaly Detection**: Multi-type detection (spikes, drift, dropouts)
- **AI-Powered Analysis**: LLM-based anomaly summarization with actionable insights
- **Robust Architecture**: Comprehensive error handling and resilience
- **Health Monitoring**: Built-in health checks and metrics
- **Scalable Design**: Docker-based microservices architecture

## 📊 Detection Capabilities

| Type | Description | Threshold | Severity |
|------|-------------|-----------|----------|
| **Temperature Drift** | Sustained high temperature | >38°C for 15s | High |
| **Pressure Spike** | Sudden pressure increase | >4.0 bar | Medium/High |
| **Flow Spike** | Excessive flow rate | >120 L/min | Medium/High |
| **Data Dropout** | Missing sensor data | >10s gap | High |

## 🏗️ Architecture
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Generator │───▶│  Redis PubSub   │───▶│Anomaly Detector │
└─────────────────┘    └─────────────────┘    └─────────────────┘
│
▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   REST API      │◀───│  LLM Summary    │◀───│  File Storage   │
└─────────────────┘    └─────────────────┘    └─────────────────┘

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 4GB+ RAM (for LLM model)
- 2GB+ disk space

### Launch the System
```bash
# Clone and start
git clone <repository>
cd water-treatment-monitoring

# Start all services
docker-compose up --build

# Check system health
curl http://localhost:8000/health
```
### Verify Operation
```bash
# View recent anomalies
curl http://localhost:8000/anomalies

# Get AI summary
curl http://localhost:8000/summary

# Check system status
curl http://localhost:8000/status

# View metrics
curl http://localhost:8000/metrics
```

Custom Configuration
# Example with custom thresholds
 -  DRIFT_THRESHOLD=35.0 
 -  PRESSURE_SPIKE_THRESHOLD=3.5 

docker-compose up

📡 API Endpoints
Core Endpoints

 - GET /health - System health check
 - GET /anomalies?limit=20&severity=high - Recent anomalies
 - GET /summary?anomaly_count=10 - AI-generated summary
 - GET /status - Component status

# Response Examples

### Health Check
```json
{
  "status": "healthy",
  "components": {
    "redis": "healthy",
    "llm": "healthy",
    "storage": "healthy"
  },
  "uptime_seconds": 1800
}
```
### Anomaly Summary
```json
{
  "summary": "Detected 3 high-severity anomalies requiring immediate attention...",
  "anomaly_count": 15,
  "status": "warning",
  "high_severity_count": 3,
  "timestamp": "2024-01-15T10:30:00Z"
}
```
