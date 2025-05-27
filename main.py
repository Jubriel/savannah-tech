# type: ignore
import json
import time
from datetime import datetime
from typing import List  # , Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
from config import settings
from utils.redis import redis_client
from utils.log import setup_logging
from llm import summarizer
from models import (
    AnomalyResponse,
    SummaryResponse,
    StatusResponse,
    HealthResponse,
)

logger = setup_logging(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Water Treatment Monitoring API",
    description=(
        "API for monitoring water treatment anomalies and "
        "generating summaries"
    ),
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
        "storage": (
            "healthy"
            if Path(settings.ANOMALY_LOG_PATH).parent.exists()
            else "unhealthy"
        )
    }

    overall_status = (
        "healthy"
        if all(status == "healthy" for status in components.values())
        else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        components=components,
        uptime_seconds=int(time.time() - startup_time)
    )


@app.get("/anomalies", response_model=List[AnomalyResponse])
async def get_anomalies(
    limit: int = Query(20, ge=1, le=100,
                       description="Number of recent anomalies to return"),
    severity: str = Query(None,
                          description="Filter by severity: low, medium, high"),
    type: str = Query(None,
                      description="Filter by type: spike, drift, dropout")
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

                    if len(anomalies) >= limit:
                        break

                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse anomaly line: {line[:50]}..."
                        f" Error: {e}"
                    )
                    continue

        logger.info(f"Retrieved {len(anomalies)} anomalies")
        return anomalies

    except Exception as e:
        logger.error(f"Error retrieving anomalies: {e}")
        raise HTTPException(status_code=500,
                            detail=f"Error retrieving anomalies: {str(e)}")


@app.get("/summary", response_model=SummaryResponse)
async def get_summary(
    anomaly_count: int = Query(
        10,
        ge=1,
        le=50,
        description="Number of recent anomalies to analyze"
    )
):
    """Generate AI-powered summary of recent anomalies"""
    try:
        summary_data = summarizer.generate_summary(anomaly_count)
        return SummaryResponse(**summary_data)

    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating summary: {str(e)}"
        )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system component status"""
    try:
        # Check individual components
        generator_status = "ok" if redis_client.is_healthy() else "error"
        detector_status = (
            "ok" if Path(settings.ANOMALY_LOG_PATH).exists() else "error"
        )
        llm_status = "ok" if summarizer.is_healthy() else "error"
        redis_status = "ok" if redis_client.is_healthy() else "error"

        # Determine overall status
        if all(
            status == "ok"
            for status in [
                generator_status,
                detector_status,
                llm_status,
                redis_status,
            ]
        ):
            overall_status = "ok"
        elif any(
            status == "ok"
            for status in [generator_status, detector_status, redis_status]
        ):
            overall_status = "degraded"
        else:
            overall_status = "error"

        return StatusResponse(
            generator=generator_status,
            detector=detector_status,
            llm=llm_status,
            redis=redis_status,
            overall=overall_status,
            timestamp=datetime.utcnow().isoformat() + 'Z'
        )

    except Exception as e:
        logger.error(f"Error checking status: {e}")
        raise HTTPException(status_code=500,
                            detail=f"Error checking status: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
