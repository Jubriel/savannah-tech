from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
from utils.log import setup_logging
from pydantic import BaseModel


logger = setup_logging(__name__)


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
