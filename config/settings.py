"""
Application settings and configuration management.
"""

from typing import Optional
from pathlib import Path
from storage.models import OptimizationConfig


# Default configuration for the application
DEFAULT_CONFIG = OptimizationConfig(
    max_phrases_stored=1000,
    generation_batch_size=15,
    iteration_delay_seconds=30,
    context_phrases_count=5,
    cleanup_threshold=10000,
    min_score_threshold=10,
    max_cpu_usage=80.0,
    max_memory_mb=2048,
    thermal_monitoring=True
)

# Default paths
DEFAULT_DB_PATH = "data/phrases.db"
DEFAULT_LOG_PATH = "logs/generator.log"

# Ollama model preferences for MacBook
PREFERRED_MODELS = [
    "llama2:7b",      # Good balance of quality and performance
    "mistral:7b",     # Fast and efficient
    "phi:2.7b",       # Very fast, good for testing
]

def get_data_dir() -> Path:
    """Get the data directory, creating it if necessary."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_log_dir() -> Path:
    """Get the logs directory, creating it if necessary."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir

def get_db_path(custom_path: Optional[str] = None) -> str:
    """Get the database path."""
    if custom_path:
        return custom_path

    data_dir = get_data_dir()
    return str(data_dir / "phrases.db")

def get_log_path(custom_path: Optional[str] = None) -> str:
    """Get the log file path."""
    if custom_path:
        return custom_path

    log_dir = get_log_dir()
    return str(log_dir / "generator.log")