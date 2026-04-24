"""Configuration management for bin_packer_3d.

Provides type-safe configuration using Pydantic models with support for
environment variables, .env files, and programmatic configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from bin_packer_3d.algorithms import ALGORITHMS


class PackerConfig(BaseModel):
    """Configuration for packing algorithms.

    Attributes:
        bin_length: Length of the bin in mm (X-axis).
        bin_width: Width of the bin in mm (Y-axis).
        bin_height: Height of the bin in mm (Z-axis).
        allow_rotation: Whether to allow box rotation.
        max_weight: Maximum weight capacity per bin in kg.
        strategy: Packing strategy — must be a key of
            :data:`bin_packer_3d.algorithms.ALGORITHMS`. Unknown values
            are rejected at construction time with an error listing the
            registered names (FR-001, FR-047).
    """

    bin_length: float = Field(default=860.0, gt=0, description="Bin length in mm")
    bin_width: float = Field(default=890.0, gt=0, description="Bin width in mm")
    bin_height: float = Field(default=1040.0, gt=0, description="Bin height in mm")
    allow_rotation: bool = Field(default=True, description="Allow box rotation")
    max_weight: float | None = Field(default=None, gt=0, description="Max weight in kg")
    strategy: str = Field(
        default="ffd",
        description="Packing strategy — validated against the ALGORITHMS registry",
    )

    @field_validator("strategy")
    @classmethod
    def _validate_strategy_registered(cls, v: str) -> str:
        """Reject strategy values not present in ALGORITHMS (FR-001, FR-047)."""
        if v not in ALGORITHMS:
            raise ValueError(f"unknown strategy {v!r}. Registered: {sorted(ALGORITHMS)}")
        return v

    @classmethod
    def get_strategies(cls) -> list[str]:
        """Return the sorted list of registered strategy names (FR-002)."""
        return sorted(ALGORITHMS)

    @property
    def bin_volume(self) -> float:
        """Calculate bin volume in cubic mm."""
        return self.bin_length * self.bin_width * self.bin_height

    @property
    def bin_volume_m3(self) -> float:
        """Calculate bin volume in cubic meters."""
        return self.bin_volume / 1_000_000_000


class VisualizationConfig(BaseModel):
    """Configuration for 3D visualization."""

    output_format: Literal["html", "png", "svg"] = Field(default="html")
    opacity: float = Field(default=0.3, ge=0, le=1)
    show_wireframe: bool = Field(default=True)
    color_by: Literal["type", "bin", "shelf"] = Field(default="type")
    auto_open: bool = Field(default=False)


class DataConfig(BaseModel):
    """Configuration for data loading."""

    input_path: Path | None = Field(default=None)
    output_dir: Path = Field(default=Path("output"))
    item_column: str = Field(default="ITEM")
    width_column: str = Field(default="W")
    height_column: str = Field(default="H")
    length_column: str = Field(default="L")
    quantity_column: str = Field(default="CANTIDAD")

    @field_validator("output_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        """Coerce a ``str`` input into a :class:`pathlib.Path` instance."""
        return Path(v) if isinstance(v, str) else v


class Settings(BaseSettings):
    """Main settings combining all configuration."""

    model_config = SettingsConfigDict(
        env_prefix="BIN_PACKER_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    packer: PackerConfig = Field(default_factory=PackerConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    debug: bool = Field(default=False)
    verbose: bool = Field(default=False)


default_settings = Settings()
