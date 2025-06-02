import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from config import settings
from utils.log import setup_logging

logger = setup_logging(__name__)


class AnomalySummarizer:
    def __init__(self):
        self.llm = None
        self.chain = None
        self.prompt = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the LLM with error handling"""
        try:
            self.llm = OllamaLLM(
                model=settings.LLM_MODEL,
                base_url=settings.LLM_HOST,
                request_timeout=1000,
                max_tokens=500
            )

            prompt_template = """
                As a water treatment system analyst.
                Analyze the following anomalies and provide a very concise
                report/summary.

                Focus on:
                1. Most critical and recent issues
                2. Patterns or trends in the anomalies

                Anomaly Data:
                {anomaly_text}

                Provide a concise and structured summary with severity levels.
                Do not Hallucinate or make up data.
            """

            self.prompt = PromptTemplate.from_template(prompt_template)
            self.chain = self.prompt | self.llm

            logger.info(f"Initialized LLM with model: {settings.LLM_MODEL}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")

    def _load_recent_anomalies(self, count: Optional[int] = None) -> List[Dict[
                                                                    str, Any]]:
        """Load recent anomalies from log file"""
        if count is None:
            count = settings.SUMMARY_ANOMALY_COUNT

        anomalies: List[Dict[str, Any]] = []
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
                            logger.warning(
                                f"Failed to parse anomaly line: {line[:50]}..."
                                f"Error: {e}"
                            )

            logger.info(f"Loaded {len(anomalies)} recent anomalies")

        except Exception as e:
            logger.error(f"Error loading anomalies: {e}")

        return anomalies

    def _format_anomalies_for_llm(self, anomalies: List[Dict[str, Any]
                                                        ]) -> str:
        """Format anomalies for LLM processing"""
        if not anomalies:
            return "No recent anomalies detected."

        formatted_lines = []

        # Group by type and severity
        by_type: dict[str, list[dict[str, Any]]] = {}
        for anomaly in anomalies:
            atype = anomaly.get('type', 'unknown')
            if atype not in by_type:
                by_type[atype] = []
            by_type[atype].append(anomaly)

        for atype, type_anomalies in by_type.items():
            formatted_lines.append(
                f"\n{atype.upper()} ANOMALIES ({len(type_anomalies)}):"
            )

            for anomaly in type_anomalies[-5:]:  # Last 5 of each type
                timestamp = anomaly.get('timestamp', 'unknown')
                parameter = anomaly.get('parameter', 'unknown')
                value = anomaly.get('value', 'unknown')
                severity = anomaly.get('severity', 'medium')
                message = anomaly.get('message', 'No message')

                formatted_lines.append(
                    f"""- {timestamp} | {parameter}: {value} |
                      Severity: {severity} | {message}"""
                )

        return "\n".join(formatted_lines)

    def generate_summary(self, anomaly_count: Optional[int] = None) -> Dict[
                                                                    str, Any]:
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
                    text = self.prompt.format(anomaly_text=anomaly_text)
                    summary_text = self.chain.invoke(input=text)

                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
                    summary_text = self._generate_fallback_summary(anomalies)
            else:
                summary_text = self._generate_fallback_summary(anomalies)

            # Determine overall status
            high_severity_count = sum(
                1 for a in anomalies if a.get('severity') == 'high'
            )
            if high_severity_count > 0:
                status = "critical" if high_severity_count > 3 else "warning"
            else:
                status = "monitoring"

            return {
                "summary": summary_text,
                "anomaly_count": len(anomalies),
                "status": status,
                # "high_severity_count": high_severity_count,
                "timestamp": anomalies[-1].get('timestamp'
                                               ) if anomalies else None
            }

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "anomaly_count": 0,
                "status": "error",
                "recommendations": ["Check system logs for detailed error \
                                     information"]
            }

    def _generate_fallback_summary(self, anomalies: List[Dict[str, Any]
                                                         ]) -> str:
        """Generate a basic summary when LLM is unavailable"""
        if not anomalies:
            return "No anomalies detected."

        # Count by type and severity
        type_counts: Dict[str, int] = {}
        severity_counts = {"high": 0, "medium": 0, "low": 0}

        for anomaly in anomalies:
            atype = anomaly.get('type', 'unknown')
            severity = anomaly.get('severity', 'medium')

            type_counts[atype] = type_counts.get(atype, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        summary_parts = [
            f"Detected {len(anomalies)} total anomalies.",
            (
                f"Severity breakdown: {severity_counts['high']} high, "
                f"{severity_counts['medium']} medium, "
                f"{severity_counts['low']} low."
            ),
            (
                "Types: "
                + ", ".join([f"{k}: {v}" for k, v in type_counts.items()])
                + "."
            )
        ]

        if severity_counts['high'] > 0:
            summary_parts.append("IMMEDIATE ATTENTION REQUIRED for \
                                    high-severity anomalies.")

        return " ".join(summary_parts)

    def is_healthy(self) -> bool:
        """Check if the summarizer is healthy"""
        return self.llm is not None and self.chain is not None


# Global summarizer instance
summarizer = AnomalySummarizer()
