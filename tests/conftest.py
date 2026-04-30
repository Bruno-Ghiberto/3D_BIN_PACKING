"""Pytest fixtures for bin_packer_3d tests."""

import pytest

from bin_packer_3d.config import PackerConfig
from bin_packer_3d.models.bin import Bin
from bin_packer_3d.models.box import Box


@pytest.fixture
def sample_box() -> Box:
    """Create a sample box for testing."""
    return Box(
        id="test_box_1",
        width=100,
        height=50,
        length=80,
        weight=1.5,
        box_type="TYPE_A",
        description="Test box",
        quantity=1,
    )


@pytest.fixture
def sample_boxes() -> list[Box]:
    """Create a list of sample boxes for testing."""
    return [
        Box(id="box_1", width=100, height=50, length=80),
        Box(id="box_2", width=200, height=100, length=150),
        Box(id="box_3", width=50, height=50, length=50),
        Box(id="box_4", width=150, height=75, length=100),
        Box(id="box_5", width=80, height=40, length=60),
    ]


@pytest.fixture
def sample_bin() -> Bin:
    """Create a sample bin for testing."""
    return Bin(
        id=1,
        length=860,
        width=890,
        height=1040,
    )


@pytest.fixture
def default_config() -> PackerConfig:
    """Create default packer configuration."""
    return PackerConfig(
        bin_length=860,
        bin_width=890,
        bin_height=1040,
    )


@pytest.fixture
def small_config() -> PackerConfig:
    """Create small bin configuration for testing edge cases."""
    return PackerConfig(
        bin_length=200,
        bin_width=200,
        bin_height=200,
    )
