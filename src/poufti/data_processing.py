import numpy as np
import pandas as pd
import json

from poufti.structures import ExperimentData

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def track_dataframe_cleanup(track_df):
    cleaned_track_df = track_df.copy()
    cleaned_track_df = cleaned_track_df.drop(columns=["weighted_cell_id", "tree_id"])
    cleaned_track_df = cleaned_track_df.sort_values(by=["track_id", "frame"])
    cleaned_track_df = cleaned_track_df.reindex(columns=[
        "track_id",
        "frame",
        "cell_id",
        "position_x",
        "position_y",
        "cellular_position_x",
        "cellular_position_y",
        "intensity"
    ])

    return cleaned_track_df

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def compile_relational_data(local_cell_registry: list, mapped_spots_df: pd.DataFrame, metadata) -> ExperimentData:
    """
    Compiles distinct cell and track DataFrames and packages them together.
    """
    cells = local_cell_registry.values()

    cell_records = []
    for c in cells:
        if not c.is_valid:
            continue

        cell_records.append({
            'phase_id': c.phase_id,
            'cell_id': c.cell_id,
            'cell_area': c.global_polygon.area if c.global_polygon else None,
            'cell_length': c.length,
            'cell_polygon_wkt': c.global_polygon.wkt if c.global_polygon else None,
            'mesh_grid': json.dumps(c.mesh_grid.tolist()) if c.mesh_grid is not None else None,
            'valid': c.is_valid
        })
        
    cells_df = pd.DataFrame(cell_records)
    
    return ExperimentData(cells=cells_df, tracks=mapped_spots_df, metadata=metadata)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------