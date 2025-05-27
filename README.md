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