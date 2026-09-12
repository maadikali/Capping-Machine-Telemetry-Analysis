"""
Central configuration loader.

Every module in this project reads paths and tunables from a single
YAML file instead of hard-coding them. Load once with `load_config()`
and pass the resulting `Settings` object down to whatever needs it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    pools_dir: str
    default_pool: str


class HeadsConfig(BaseModel):
    ids: List[str]


class SchemaConfig(BaseModel):
    timestamp_col: str = "timestamp"
    torque_suffix: str = " AppTorque"
    status_suffix: str = " Status"
    count_suffix: str = " Count"


class AnalyticsConfig(BaseModel):
    torque_expected_range_nm: List[float]
    moving_average_window: int = 50
    drift_zscore_threshold: float = 3.0
    idle_no_load_seconds: int = 120
    correlation_min_events: int = 30


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "logs"


class Settings(BaseModel):
    data: DataConfig
    heads: HeadsConfig
    schema_: SchemaConfig = Field(alias="schema")
    analytics: AnalyticsConfig
    logging: LoggingConfig

    project_root: Path = Field(default_factory=Path.cwd, exclude=True)

    model_config = ConfigDict(populate_by_name=True)

    def resolve(self, relative_path: str) -> Path:
        """Resolve a config-relative path against the project root."""
        p = Path(relative_path)
        return p if p.is_absolute() else (self.project_root / p)

    @property
    def pools_dir(self) -> Path:
        return self.resolve(self.data.pools_dir)


def load_config(config_path: str | None = None) -> Settings:
    """
    Load settings from YAML.

    Resolution order:
      1. explicit `config_path` argument
      2. TELEMETRY_CONFIG environment variable
      3. ./config/config.yaml relative to the current working directory
    """
    path = (
        config_path
        or os.environ.get("TELEMETRY_CONFIG")
        or "config/config.yaml"
    )
    config_file = Path(path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found at '{config_file}'. "
            "Set TELEMETRY_CONFIG or pass config_path to load_config()."
        )

    with open(config_file, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    settings = Settings(**raw)
    settings.project_root = config_file.resolve().parent.parent
    return settings
