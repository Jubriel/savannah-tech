# type: ignore
import random
import time
import datetime
from config import settings
from utils.redis import redis_client
from utils.log import setup_logging
from detector import SensorReading


logger = setup_logging(__name__)


class SensorDataGenerator:
    def __init__(self):
        self.sensor_id = settings.SENSOR_ID
        self.running = False

    def generate_reading(self) -> SensorReading:
        """Generate realistic sensor reading with some variability"""
        return SensorReading(
            timestamp=datetime.datetime.now().isoformat() + 'Z',
            sensor_id=self.sensor_id,
            temperature=round(random.uniform(10, 40), 1),
            pressure=round(random.uniform(1.0, 4.0), 2),
            flow=round(random.uniform(20, 130), 1)
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
                    logger.warning(
                        (
                            f"Failed to publish reading "
                            f"(failure {consecutive_failures})"
                        )
                    )

                if consecutive_failures >= max_failures:
                    logger.error(
                        (
                            f"Too many consecutive failures ({max_failures}), "
                            "stopping"
                        )
                    )
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
