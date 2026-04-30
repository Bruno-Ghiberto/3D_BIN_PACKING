# 3D Bin Packing Solver

[![CI](https://github.com/Bruno-Ghiberto/3D_BIN_PACKING/actions/workflows/ci.yml/badge.svg)](https://github.com/Bruno-Ghiberto/3D_BIN_PACKING/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Bruno-Ghiberto/3D_BIN_PACKING/branch/main/graph/badge.svg?flag=library)](https://codecov.io/gh/Bruno-Ghiberto/3D_BIN_PACKING)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A professional Python implementation for solving the **3D Bin Packing Problem (3D-BPP)**, an NP-hard optimization problem with applications in logistics, warehousing, and container loading.

## Features

- **Multiple Packing Algorithms**: First-Fit Decreasing (FFD) and Shelf-based packing
- **6-Axis Box Rotation**: Automatic orientation optimization for better space utilization
- **Interactive 3D Visualization**: Plotly-powered visualizations with color-coded boxes
- **Professional CLI**: Easy-to-use command-line interface with Rich formatting
- **Type-Safe Design**: Full type hints with Pydantic configuration validation
- **Comprehensive Metrics**: Utilization %, success rate, timing, and per-bin statistics
- **Flexible Data Loading**: CSV and Excel file support

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Bruno-Ghiberto/3D_BIN_PACKING.git
cd 3D_BIN_PACKING

# Install the package
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Pack boxes from a CSV file
bin-packer pack data.csv --strategy ffd --visualize

# View available algorithms
bin-packer info

# Initialize configuration file
bin-packer init
```

### Python API

```python
from bin_packer_3d import Box, PackerConfig
from bin_packer_3d.algorithms import FirstFitDecreasingPacker
from bin_packer_3d.visualization import Plotter3D

# Create boxes
boxes = [
    Box(id="B001", width=200, height=150, length=100),
    Box(id="B002", width=300, height=200, length=150),
    Box(id="B003", width=150, height=100, length=80),
]

# Configure packer
config = PackerConfig(
    bin_length=860,
    bin_width=890,
    bin_height=1040,
)

# Pack boxes
packer = FirstFitDecreasingPacker(config)
result = packer.pack(boxes)

# Display results
print(f"Boxes placed: {result.placed_count}/{result.total_boxes}")
print(f"Bins used: {result.bins_used}")
print(f"Utilization: {result.utilization_percent:.1f}%")

# Generate visualization
plotter = Plotter3D()
plotter.plot_result(result, output_dir="output/")
```

## Project Structure

```
3D_BIN_PACKING/
├── src/bin_packer_3d/       # Main package
│   ├── algorithms/          # Packing algorithms (FFD, Shelf)
│   ├── models/              # Data models (Box, Bin, Placement)
│   ├── visualization/       # 3D Plotly visualization
│   ├── data/                # CSV/Excel loaders
│   ├── utils/               # Metrics calculation
│   ├── cli.py               # Command-line interface
│   └── config.py            # Pydantic configuration
├── tests/                   # Pytest test suite
├── DATASETS/                # Sample data files
├── CODE/                    # Legacy implementation (reference)
└── pyproject.toml           # Modern Python packaging
```

## Algorithms

### First-Fit Decreasing (FFD)

Classic heuristic that sorts boxes by volume (largest first) and places each box in the first bin that has space.

- **Time Complexity**: O(n log n) for sorting + O(n × m) for placement
- **Best for**: General-purpose packing with good average performance

### Shelf-Based Packing

Extends 2D shelf packing to 3D by organizing boxes on horizontal "shelves" within each bin, using guillotine cuts for space splitting.

- **Time Complexity**: O(n × s × r) where s = shelves, r = rectangles
- **Best for**: Scenarios with boxes of similar heights

## Input Format

CSV/Excel files should have these columns:

| Column | Description | Required |
|--------|-------------|----------|
| `ITEM` | Unique box identifier | Yes |
| `W` | Width in mm | Yes |
| `H` | Height in mm | Yes |
| `L` | Length in mm | Yes |
| `CANTIDAD` | Quantity (creates multiple boxes) | No (default: 1) |
| `CAJA` | Box type for color coding | No |
| `DESCRIPCION` | Description for hover text | No |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=bin_packer_3d --cov-report=html

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v
```

**Current Status**: 39 tests passing

## Configuration

Settings can be configured via:
- Command-line arguments
- Environment variables (prefix: `BIN_PACKER_`)
- `.env` file

```bash
# Generate default configuration
bin-packer init
```

## Problem Background

The 3D Bin Packing Problem is a classic NP-hard combinatorial optimization problem. Given a set of rectangular boxes and bins of fixed dimensions, the goal is to pack all boxes into the minimum number of bins while respecting:

- **Geometric constraints**: Boxes must fit within bin boundaries
- **Non-overlap constraint**: Boxes cannot intersect
- **Orientation options**: Boxes can be rotated in 6 ways

This implementation uses heuristic approaches that provide good solutions in polynomial time, making it practical for real-world logistics applications.

## Dependencies

- **pandas** >= 2.0.0: Data manipulation
- **numpy** >= 1.24.0: Numerical operations
- **plotly** >= 5.18.0: Interactive 3D visualization
- **click** >= 8.1.0: CLI framework
- **pydantic** >= 2.0.0: Configuration validation
- **rich** >= 13.0.0: Terminal formatting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Bruno Ghiberto - [GitHub Profile](https://github.com/Bruno-Ghiberto)

---

*This project demonstrates solving a classic NP-hard optimization problem using heuristic algorithms with professional Python engineering practices.*
