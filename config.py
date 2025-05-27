import os
from typing import Dict


class Settings:
    # Redis Configuration
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'redis')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB: int = int(os.getenv('REDIS_DB', '0'))
    REDIS_RETRY_ATTEMPTS: int = int(os.getenv('REDIS_RETRY_ATTEMPTS', '3'))
    REDIS_RETRY_DELAY: int = int(os.getenv('REDIS_RETRY_DELAY', '5'))

    # Sensor Configuration
    SENSOR_ID: str = os.getenv('SENSOR_ID', 'wtf-pipe-1')
    DATA_GENERATION_INTERVAL: int = int(os.getenv('DATA_GENERATION_INTERVAL',
                                                  '2'))

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
    ANOMALY_LOG_PATH: str = os.getenv('ANOMALY_LOG_PATH', 'anomaly_log.json')
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '5'))

    # LLM Configuration
    LLM_MODEL: str = os.getenv('LLM_MODEL', 'llama3.2')
    LLM_HOST: str = os.getenv('LLM_HOST', 'http://localhost:11434')
    SUMMARY_ANOMALY_COUNT: int = int(os.getenv('SUMMARY_ANOMALY_COUNT', '10'))

    # API Configuration
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))

    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')


settings = Settings()
