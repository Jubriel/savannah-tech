# Optimized Water Treatment Monitoring System

## Key Improvements Made

### 1. **Enhanced Error Handling & Resilience**
- Added comprehensive exception handling throughout all components
- Implemented connection retry logic for Redis and external services
- Graceful degradation when services are unavailable

### 2. **Performance Optimizations**
- Batch processing for anomaly storage
- Connection pooling for Redis
- Async processing in API layer
- Optimized file I/O operations

### 3. **Code Structure & Maintainability**
- Separated configuration into environment variables
- Added proper logging throughout the system
- Improved type hints and documentation
- Modular design with clear separation of concerns

### 4. **Monitoring & Observability**
- Health check endpoints with detailed status
- Structured logging with different levels
- Metrics collection ready integration points

---

## Optimized File Structure

```
water-treatment-monitoring/
├── config/
│   ├── __init__.py
│   └── settings.py
├── data_generator/
│   ├── __init__.py
│   └── generator.py
├── anomaly_detector/
│   ├── __init__.py
│   ├── detector.py
│   └── models.py
├── llm_summary/
│   ├── __init__.py
│   └── summarizer.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── models.py
├── storage/
│   └── anomaly_log.json
├── utils/
│   ├── __init__.py
│   ├── redis_client.py
│   └── logging_config.py
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.llm
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Optimized Code Files

### config/settings.py
```python
import os
from typing import Dict, Any

class Settings:
    # Redis Configuration
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'redis')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB: int = int(os.getenv('REDIS_DB', '0'))
    REDIS_RETRY_ATTEMPTS: int = int(os.getenv('REDIS_RETRY_ATTEMPTS', '3'))
    REDIS_RETRY_DELAY: int = int(os.getenv('REDIS_RETRY_DELAY', '5'))
    
    # Sensor Configuration
    SENSOR_ID: str = os.getenv('SENSOR_ID', 'wtf-pipe-1')
    DATA_GENERATION_INTERVAL: int = int(os.getenv('DATA_GENERATION_INTERVAL', '2'))
    
    # Anomaly Detection Thresholds
    WINDOW_SIZE: int = int(os.getenv('WINDOW_SIZE', '10'))
    DRIFT_THRESHOLD: float = float(os.getenv('DRIFT_THRESHOLD', '38.0'))
    SPIKE_THRESHOLDS: Dict[str, float] = {
        'pressure': float(os.getenv('PRESSURE_SPIKE_THRESHOLD', '4.0')),
        'flow': float(os.getenv('FLOW_SPIKE_THRESHOLD', '120.0')),
        'temperature': float(os.getenv('TEMP_SPIKE_THRESHOLD', '45.0'))
    }
    DROPOUT_SECONDS: int = int(os.getenv('DROPOUT_SECONDS', '10'))
    
    # Storage Configuration
    ANOMALY_LOG_PATH: str = os.getenv('ANOMALY_LOG_PATH', '/app/storage/anomaly_log.json')
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '5'))
    
    # LLM Configuration
    LLM_MODEL: str = os.getenv('LLM_MODEL', 'mistral')
    LLM_HOST: str = os.getenv('LLM_HOST', 'http://ollama:11434')
    SUMMARY_ANOMALY_COUNT: int = int(os.getenv('SUMMARY_ANOMALY_COUNT', '10'))
    
    # API Configuration
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')

settings = Settings()
```

### utils/logging_config.py
```python
import logging
import sys
from config.settings import settings

def setup_logging(name: str) -> logging.Logger:
    """Setup structured logging for the application"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
```

### utils/redis_client.py
```python
import redis
import json
import time
from typing import Optional, Any
from config.settings import settings
from utils.logging_config import setup_logging

logger = setup_logging(__name__)

class RedisClient:
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self._connect()
    
    def _connect(self) -> bool:
        """Establish Redis connection with retry logic"""
        for attempt in range(settings.REDIS_RETRY_ATTEMPTS):
            try:
                self.client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                self.client.ping()
                logger.info(f"Connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
                return True
            except Exception as e:
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                if attempt < settings.REDIS_RETRY_ATTEMPTS - 1:
                    time.sleep(settings.REDIS_RETRY_DELAY)
        
        logger.error("Failed to connect to Redis after all attempts")
        return False
    
    def publish(self, channel: str, data: Any) -> bool:
        """Publish data to Redis channel"""
        try:
            if not self.client:
                if not self._connect():
                    return False
            
            message = json.dumps(data) if not isinstance(data, str) else data
            self.client.publish(channel, message)
            return True
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}")
            self.client = None  # Force reconnection on next attempt
            return False
    
    def subscribe(self, channel: str) -> Optional[redis.client.PubSub]:
        """Subscribe to Redis channel"""
        try:
            if not self.client:
                if not self._connect():
                    return None
            
            self.pubsub = self.client.pubsub()
            self.pubsub.subscribe(channel)
            logger.info(f"Subscribed to channel: {channel}")
            return self.pubsub
        except Exception as e:
            logger.error(f"Failed to subscribe to {channel}: {e}")
            return None
    
    def is_healthy(self) -> bool:
        """Check if Redis connection is healthy"""
        try:
            return self.client is not None and self.client.ping()
        except:
            return False

# Global Redis client instance
redis_client = RedisClient()
```

### anomaly_detector/models.py
```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

@dataclass
class SensorReading:
    timestamp: str
    sensor_id: str
    temperature: float
    pressure: float
    flow: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorReading':
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'sensor_id': self.sensor_id,
            'temperature': self.temperature,
            'pressure': self.pressure,
            'flow': self.flow
        }

@dataclass
class Anomaly:
    type: str
    timestamp: str
    sensor_id: str
    parameter: str
    value: float
    message: str
    duration_seconds: Optional[int] = None
    severity: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'type': self.type,
            'timestamp': self.timestamp,
            'sensor_id': self.sensor_id,
            'parameter': self.parameter,
            'value': self.value,
            'message': self.message,
            'severity': self.severity
        }
        if self.duration_seconds is not None:
            result['duration_seconds'] = self.duration_seconds
        return result
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
```

### data_generator/generator.py
```python
import json
import random
import time
import datetime
from typing import Dict, Any
from config.settings import settings
from utils.redis_client import redis_client
from utils.logging_config import setup_logging
from anomaly_detector.models import SensorReading

logger = setup_logging(__name__)

class SensorDataGenerator:
    def __init__(self):
        self.sensor_id = settings.SENSOR_ID
        self.running = False
    
    def generate_reading(self) -> SensorReading:
        """Generate realistic sensor reading with some variability"""
        # Add some realistic patterns and occasional anomalies for testing
        base_temp = 25.0 + random.uniform(-5, 5)
        base_pressure = 2.0 + random.uniform(-0.5, 0.5)
        base_flow = 60.0 + random.uniform(-20, 20)
        
        # Occasionally inject test anomalies (1% chance)
        if random.random() < 0.01:
            if random.choice([True, False]):
                base_pressure = random.uniform(4.5, 6.0)  # Pressure spike
            else:
                base_temp = random.uniform(39, 42)  # Temperature drift
        
        return SensorReading(
            timestamp=datetime.datetime.utcnow().isoformat() + 'Z',
            sensor_id=self.sensor_id,
            temperature=round(base_temp, 1),
            pressure=round(base_pressure, 2),
            flow=round(base_flow, 1)
        )
    
    def start_generation(self):
        """Start the data generation loop"""
        self.running = True
        logger.info(f"Starting sensor data generation for {self.sensor_id}")
        
        consecutive_failures = 0
        max_failures = 5
        
        while self.running:
            try:
                reading = self.generate_reading()
                
                if redis_client.publish('sensor_data', reading.to_dict()):
                    consecutive_failures = 0
                    logger.debug(f"Published reading: {reading.to_dict()}")
                else:
                    consecutive_failures += 1
                    logger.warning(f"Failed to publish reading (failure {consecutive_failures})")
                
                if consecutive_failures >= max_failures:
                    logger.error(f"Too many consecutive failures ({max_failures}), stopping")
                    break
                
                time.sleep(settings.DATA_GENERATION_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping generator")
                break
            except Exception as e:
                logger.error(f"Unexpected error in data generation: {e}")
                consecutive_failures += 1
                time.sleep(settings.DATA_GENERATION_INTERVAL)
        
        self.running = False
        logger.info("Data generation stopped")
    
    def stop(self):
        """Stop the data generation"""
        self.running = False

if __name__ == "__main__":
    generator = SensorDataGenerator()
    try:
        generator.start_generation()
    except KeyboardInterrupt:
        generator.stop()
```

### anomaly_detector/detector.py
```python
import json
import time
import os
from collections import deque
from typing import List, Dict, Any, Optional
from pathlib import Path
from config.settings import settings
from utils.redis_client import redis_client
from utils.logging_config import setup_logging
from anomaly_detector.models import SensorReading, Anomaly

logger = setup_logging(__name__)

class AnomalyDetector:
    def __init__(self):
        self.window = deque(maxlen=settings.WINDOW_SIZE)
        self.last_timestamp = time.time()
        self.anomaly_buffer: List[Anomaly] = []
        self.drift_start_time: Optional[float] = None
        
        # Ensure storage directory exists
        storage_path = Path(settings.ANOMALY_LOG_PATH).parent
        storage_path.mkdir(parents=True, exist_ok=True)
    
    def detect_spikes(self, reading: SensorReading) -> List[Anomaly]:
        """Detect parameter spikes"""
        anomalies = []
        
        for param, threshold in settings.SPIKE_THRESHOLDS.items():
            value = getattr(reading, param)
            if value > threshold:
                severity = "high" if value > threshold * 1.5 else "medium"
                anomalies.append(Anomaly(
                    type="spike",
                    timestamp=reading.timestamp,
                    sensor_id=reading.sensor_id,
                    parameter=param,
                    value=value,
                    message=f"{param.title()} spike detected: {value} (threshold: {threshold})",
                    severity=severity
                ))
        
        return anomalies
    
    def detect_drift(self, reading: SensorReading) -> List[Anomaly]:
        """Detect temperature drift over time"""
        anomalies = []
        self.window.append(reading)
        
        if len(self.window) == self.window.maxlen:
            temps = [r.temperature for r in self.window]
            
            if all(t > settings.DRIFT_THRESHOLD for t in temps):
                if self.drift_start_time is None:
                    self.drift_start_time = time.time()
                
                drift_duration = time.time() - self.drift_start_time
                if drift_duration >= 15:  # 15 seconds of drift
                    anomalies.append(Anomaly(
                        type="drift",
                        timestamp=reading.timestamp,
                        sensor_id=reading.sensor_id,
                        parameter="temperature",
                        value=reading.temperature,
                        duration_seconds=int(drift_duration),
                        message=f"Temperature drift detected over {int(drift_duration)} seconds. "
                               f"Average: {sum(temps)/len(temps):.1f}°C",
                        severity="high"
                    ))
                    self.drift_start_time = None
                    self.window.clear()
            else:
                self.drift_start_time = None
        
        return anomalies
    
    def detect_dropout(self) -> List[Anomaly]:
        """Detect data dropouts (no data received)"""
        anomalies = []
        current_time = time.time()
        
        if current_time - self.last_timestamp > settings.DROPOUT_SECONDS:
            anomalies.append(Anomaly(
                type="dropout",
                timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                sensor_id=settings.SENSOR_ID,
                parameter="data_flow",
                value=current_time - self.last_timestamp,
                message=f"Data dropout detected: No data for {int(current_time - self.last_timestamp)} seconds",
                severity="high"
            ))
        
        return anomalies
    
    def store_anomalies(self, anomalies: List[Anomaly]):
        """Store anomalies with batch processing"""
        if not anomalies:
            return
        
        self.anomaly_buffer.extend(anomalies)
        
        if len(self.anomaly_buffer) >= settings.BATCH_SIZE:
            self._flush_anomalies()
    
    def _flush_anomalies(self):
        """Flush anomaly buffer to file"""
        if not self.anomaly_buffer:
            return
        
        try:
            with open(settings.ANOMALY_LOG_PATH, "a") as f:
                for anomaly in self.anomaly_buffer:
                    f.write(anomaly.to_json() + "\n")
            
            logger.info(f"Stored {len(self.anomaly_buffer)} anomalies")
            self.anomaly_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to store anomalies: {e}")
    
    def process_reading(self, reading: SensorReading):
        """Process a sensor reading for anomalies"""
        self.last_timestamp = time.time()
        
        all_anomalies = []
        all_anomalies.extend(self.detect_spikes(reading))
        all_anomalies.extend(self.detect_drift(reading))
        
        if all_anomalies:
            logger.warning(f"Detected {len(all_anomalies)} anomalies in reading from {reading.sensor_id}")
            for anomaly in all_anomalies:
                logger.warning(f"  - {anomaly.message}")
        
        self.store_anomalies(all_anomalies)
    
    def start_detection(self):
        """Start the anomaly detection loop"""
        logger.info("Starting anomaly detection")
        
        pubsub = redis_client.subscribe('sensor_data')
        if not pubsub:
            logger.error("Failed to subscribe to sensor data, exiting")
            return
        
        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        reading = SensorReading.from_dict(data)
                        self.process_reading(reading)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                
                # Check for dropouts periodically
                dropout_anomalies = self.detect_dropout()
                if dropout_anomalies:
                    self.store_anomalies(dropout_anomalies)
        
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping detector")
        except Exception as e:
            logger.error(f"Unexpected error in anomaly detection: {e}")
        finally:
            self._flush_anomalies()  # Flush remaining anomalies
            if pubsub:
                pubsub.close()

if __name__ == "__main__":
    detector = AnomalyDetector()
    detector.start_detection()
```

### llm_summary/summarizer.py
```python
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from config.settings import settings
from utils.logging_config import setup_logging

logger = setup_logging(__name__)

class AnomalySummarizer:
    def __init__(self):
        self.llm = None
        self.chain = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM with error handling"""
        try:
            self.llm = Ollama(
                model=settings.LLM_MODEL,
                base_url=settings.LLM_HOST
            )
            
            prompt_template = """
You are a water treatment system analyst. Analyze the following anomalies and provide a concise summary.

Focus on:
1. Most critical issues requiring immediate attention
2. Patterns or trends in the anomalies
3. Recommended actions

Anomaly Data:
{anomaly_text}

Provide a structured summary with severity levels and actionable insights.
"""
            
            prompt = PromptTemplate.from_template(prompt_template)
            self.chain = LLMChain(llm=self.llm, prompt=prompt)
            
            logger.info(f"Initialized LLM with model: {settings.LLM_MODEL}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
    
    def _load_recent_anomalies(self, count: int = None) -> List[Dict[str, Any]]:
        """Load recent anomalies from log file"""
        if count is None:
            count = settings.SUMMARY_ANOMALY_COUNT
        
        anomalies = []
        log_path = Path(settings.ANOMALY_LOG_PATH)
        
        if not log_path.exists():
            logger.warning(f"Anomaly log file not found: {log_path}")
            return anomalies
        
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                # Get the last 'count' lines
                recent_lines = lines[-count:] if lines else []
                
                for line in recent_lines:
                    line = line.strip()
                    if line:
                        try:
                            anomaly = json.loads(line)
                            anomalies.append(anomaly)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse anomaly line: {line[:50]}... Error: {e}")
            
            logger.info(f"Loaded {len(anomalies)} recent anomalies")
            
        except Exception as e:
            logger.error(f"Error loading anomalies: {e}")
        
        return anomalies
    
    def _format_anomalies_for_llm(self, anomalies: List[Dict[str, Any]]) -> str:
        """Format anomalies for LLM processing"""
        if not anomalies:
            return "No recent anomalies detected."
        
        formatted_lines = []
        
        # Group by type and severity
        by_type = {}
        for anomaly in anomalies:
            atype = anomaly.get('type', 'unknown')
            if atype not in by_type:
                by_type[atype] = []
            by_type[atype].append(anomaly)
        
        for atype, type_anomalies in by_type.items():
            formatted_lines.append(f"\n{atype.upper()} ANOMALIES ({len(type_anomalies)}):")
            
            for anomaly in type_anomalies[-5:]:  # Last 5 of each type
                timestamp = anomaly.get('timestamp', 'unknown')
                parameter = anomaly.get('parameter', 'unknown')
                value = anomaly.get('value', 'unknown')
                severity = anomaly.get('severity', 'medium')
                message = anomaly.get('message', 'No message')
                
                formatted_lines.append(
                    f"- {timestamp} | {parameter}: {value} | Severity: {severity} | {message}"
                )
        
        return "\n".join(formatted_lines)
    
    def generate_summary(self, anomaly_count: Optional[int] = None) -> Dict[str, Any]:
        """Generate AI-powered summary of recent anomalies"""
        try:
            # Load recent anomalies
            anomalies = self._load_recent_anomalies(anomaly_count)
            
            if not anomalies:
                return {
                    "summary": "No recent anomalies to analyze.",
                    "anomaly_count": 0,
                    "status": "healthy",
                    "recommendations": []
                }
            
            # Format for LLM
            anomaly_text = self._format_anomalies_for_llm(anomalies)
            
            # Generate summary using LLM
            if self.chain:
                try:
                    llm_response = self.chain.run(anomaly_text=anomaly_text)
                    summary_text = llm_response.strip()
                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
                    summary_text = self._generate_fallback_summary(anomalies)
            else:
                summary_text = self._generate_fallback_summary(anomalies)
            
            # Determine overall status
            high_severity_count = sum(1 for a in anomalies if a.get('severity') == 'high')
            if high_severity_count > 0:
                status = "critical" if high_severity_count > 3 else "warning"
            else:
                status = "monitoring"
            
            return {
                "summary": summary_text,
                "anomaly_count": len(anomalies),
                "status": status,
                "high_severity_count": high_severity_count,
                "timestamp": anomalies[-1].get('timestamp') if anomalies else None
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "anomaly_count": 0,
                "status": "error",
                "recommendations": ["Check system logs for detailed error information"]
            }
    
    def _generate_fallback_summary(self, anomalies: List[Dict[str, Any]]) -> str:
        """Generate a basic summary when LLM is unavailable"""
        if not anomalies:
            return "No anomalies detected."
        
        # Count by type and severity
        type_counts = {}
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        
        for anomaly in anomalies:
            atype = anomaly.get('type', 'unknown')
            severity = anomaly.get('severity', 'medium')
            
            type_counts[atype] = type_counts.get(atype, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        summary_parts = [
            f"Detected {len(anomalies)} total anomalies.",
            f"Severity breakdown: {severity_counts['high']} high, {severity_counts['medium']} medium, {severity_counts['low']} low.",
            f"Types: {', '.join([f'{k}: {v}' for k, v in type_counts.items()])}."
        ]
        
        if severity_counts['high'] > 0:
            summary_parts.append("IMMEDIATE ATTENTION REQUIRED for high-severity anomalies.")
        
        return " ".join(summary_parts)
    
    def is_healthy(self) -> bool:
        """Check if the summarizer is healthy"""
        return self.llm is not None and self.chain is not None

# Global summarizer instance
summarizer = AnomalySummarizer()
```

### api/models.py
```python
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class AnomalyResponse(BaseModel):
    type: str
    timestamp: str
    sensor_id: str
    parameter: str
    value: float
    message: str
    severity: str = "medium"
    duration_seconds: Optional[int] = None

class SummaryResponse(BaseModel):
    summary: str
    anomaly_count: int
    status: str
    high_severity_count: int = 0
    timestamp: Optional[str] = None

class StatusResponse(BaseModel):
    generator: str
    detector: str
    llm: str
    redis: str
    overall: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str]
    uptime_seconds: int
```

### api/main.py
```python
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config.settings import settings
from utils.redis_client import redis_client
from utils.logging_config import setup_logging
from llm_summary.summarizer import summarizer
from api.models import AnomalyResponse, SummaryResponse, StatusResponse, HealthResponse

logger = setup_logging(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Water Treatment Monitoring API",
    description="API for monitoring water treatment anomalies and generating summaries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track startup time for uptime calculation
startup_time = time.time()

@app.get("/health", response_model=HealthResponse)
async def get_health():
    """Comprehensive health check endpoint"""
    components = {
        "redis": "healthy" if redis_client.is_healthy() else "unhealthy",
        "llm": "healthy" if summarizer.is_healthy() else "unhealthy",
        "storage": "healthy" if Path(settings.ANOMALY_LOG_PATH).parent.exists() else "unhealthy"
    }
    
    overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        components=components,
        uptime_seconds=int(time.time() - startup_time)
    )

@app.get("/anomalies", response_model=List[AnomalyResponse])
async def get_anomalies(
    limit: int = Query(20, ge=1, le=100, description="Number of recent anomalies to return"),
    severity: str = Query(None, description="Filter by severity: low, medium, high"),
    type: str = Query(None, description="Filter by type: spike, drift, dropout")
):
    """Get recent anomalies with optional filtering"""
    try:
        log_path = Path(settings.ANOMALY_LOG_PATH)
        
        if not log_path.exists():
            logger.warning(f"Anomaly log file not found: {log_path}")
            return []
        
        anomalies = []
        
        with open(log_path, 'r') as f:
            lines = f.readlines()
            
            # Process lines in reverse order (most recent first)
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    anomaly_data = json.loads(line)
                    
                    # Apply filters
                    if severity and anomaly_data.get('severity') != severity:
                        continue
                    if type and anomaly_data.get('type') != type:
                        continue
                    
                    anomalies.append(AnomalyResponse(**anomaly_data))