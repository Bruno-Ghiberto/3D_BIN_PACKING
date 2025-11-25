"""Data loading utilities for CSV and Excel files.

This module provides functions to load box data from various file formats
into Box objects ready for packing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bin_packer_3d.models.box import Box
from bin_packer_3d.config import DataConfig, default_settings


def load_boxes_from_csv(
    file_path: Path | str,
    config: DataConfig | None = None,
) -> list[Box]:
    """Load boxes from a CSV file.
    
    Args:
        file_path: Path to CSV file.
        config: Data configuration for column mappings.
    
    Returns:
        List of Box objects.
    
    Example:
        >>> boxes = load_boxes_from_csv("data.csv")
        >>> print(f"Loaded {len(boxes)} boxes")
    """
    config = config or default_settings.data
    df = pd.read_csv(file_path)
    return _dataframe_to_boxes(df, config)


def load_boxes_from_excel(
    file_path: Path | str,
    sheet_name: str | int = 0,
    config: DataConfig | None = None,
) -> list[Box]:
    """Load boxes from an Excel file.
    
    Args:
        file_path: Path to Excel file.
        sheet_name: Sheet name or index to load.
        config: Data configuration for column mappings.
    
    Returns:
        List of Box objects.
    """
    config = config or default_settings.data
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return _dataframe_to_boxes(df, config)


def _dataframe_to_boxes(df: pd.DataFrame, config: DataConfig) -> list[Box]:
    """Convert DataFrame rows to Box objects.
    
    Handles quantity column to create multiple boxes when needed.
    """
    # Clean column names
    df.columns = df.columns.str.strip()
    
    boxes: list[Box] = []
    
    for _, row in df.iterrows():
        try:
            # Get dimensions
            width = float(row[config.width_column])
            height = float(row[config.height_column])
            length = float(row[config.length_column])
            
            # Get item ID
            item_id = str(row[config.item_column])
            
            # Get quantity (default to 1)
            quantity = 1
            if config.quantity_column in row.index:
                qty_val = row[config.quantity_column]
                if pd.notna(qty_val):
                    quantity = int(qty_val)
            
            # Get optional fields
            box_type = ""
            description = ""
            weight = 0.0
            
            if "CAJA" in row.index and pd.notna(row["CAJA"]):
                box_type = str(row["CAJA"])
            if "DESCRIPCION" in row.index and pd.notna(row["DESCRIPCION"]):
                description = str(row["DESCRIPCION"])
            if "PESO" in row.index and pd.notna(row["PESO"]):
                weight = float(row["PESO"])
            
            # Create boxes (one per quantity)
            for i in range(quantity):
                box = Box(
                    id=f"{item_id}_{i+1}" if quantity > 1 else item_id,
                    width=width,
                    height=height,
                    length=length,
                    weight=weight,
                    box_type=box_type,
                    description=description,
                    quantity=1,  # Individual box
                )
                boxes.append(box)
                
        except (KeyError, ValueError) as e:
            print(f"Warning: Skipping row due to error: {e}")
            continue
    
    return boxes


def save_placements_to_csv(
    result: "PlacementResult",
    output_path: Path | str,
) -> None:
    """Save packing result to CSV file.
    
    Args:
        result: PlacementResult to save.
        output_path: Path for output CSV.
    """
    from bin_packer_3d.models.placement import PlacementResult
    
    records = [p.to_dict() for p in result.all_placements()]
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(records)} placements to: {output_path}")
