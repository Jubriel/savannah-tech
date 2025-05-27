# type: ignore
from typing import List, Optional
import json
import time
from collections import deque
from pathlib import Path
from config import settings
from utils.redis import redis_client
from utils.log import setup_logging
from models import SensorReading, Anomaly

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
                    message=f"""{param.title()} spike detected: {value}
                                 (threshold: {threshold})""",
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
                        message=(
                            f""""Temperature drift detected over {
                                int(drift_duration)
                                } seconds."""
                            f"Average: {sum(temps)/len(temps):.1f}°C"
                        ),
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
                message=f"""Data dropout detected: No data for {
                            int(current_time - self.last_timestamp)
                            } seconds""",
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
            logger.warning(f"Detected {len(all_anomalies)} \
                           anomalies in reading from {reading.sensor_id}")
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
