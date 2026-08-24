import spotiflow
import numpy as np
from os import path
from pathlib import Path
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from laptrack import LapTrack
from laptrack import datasets
import shapely

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def filter_and_map_spots_single_frame(cells: list, spots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps a single-frame DataFrame of loci to a list of cells.
    Spots not contained within any cell are automatically dropped.
    
    Expected spots_df columns: ['position_x', 'position_y']
    """
    # 1. Work on a copy to prevent SettingWithCopyWarnings
    df = spots_df.copy()
    df['cell_id'] = -1 
    
    # 2. Extract valid cells
    valid_cells = [c for c in cells if c.is_valid and c.global_polygon is not None]
    
    # If no cells exist or the dataframe is empty, return an empty dataframe immediately
    if not valid_cells or df.empty:
        return df[df['cell_id'] != -1].copy()
        
    polygons = [c.global_polygon for c in valid_cells]
    cell_ids = np.array([c.cell_id for c in valid_cells])
    
    # 3. Build the spatial index
    tree = shapely.STRtree(polygons)
    
    # 4. Extract coordinates and query the tree
    loci_xy_array = df[['position_x', 'position_y']].to_numpy()
    loci_points = shapely.points(loci_xy_array)
    
    locus_indices, poly_indices = tree.query(loci_points, predicate="within")
    
    # 5. Map the results back to the original DataFrame
    # locus_indices correspond to the row order, but we must use the true Pandas index
    original_indices = df.index[locus_indices]
    matched_cell_ids = cell_ids[poly_indices]
    
    # Drop the matched IDs into the exact correct rows
    df.loc[original_indices, 'cell_id'] = matched_cell_ids
        
    # 6. Filter out all background spots at once (where cell_id is still -1)
    filtered_df = df[df['cell_id'] != -1].copy()
    filtered_df['cell_id'] = filtered_df['cell_id'].astype(int)
    
    return filtered_df

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_spotiflow(tracking_array, model, local_cell_registry, prob_thresh):
    spots_df = pd.DataFrame()
    cells = local_cell_registry.values()

    for idx, frame in enumerate(tracking_array, start=1):
        points, details = model.predict(frame, subpix=True, prob_thresh=prob_thresh, verbose=False)

        if len(points) == 0:
            continue
            
        frame_df = pd.DataFrame({
            "frame": idx,
            "position_x": tuple(points[:,1]),
            "position_y": tuple(points[:,0]),
            "intensity": tuple(map(float, details.intens))
        })
        
        cleaned_frame_df = filter_and_map_spots_single_frame(cells, frame_df)

        if not cleaned_frame_df.empty:
            spots_df = pd.concat([spots_df, cleaned_frame_df], ignore_index=True)
            
    return spots_df

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def initialize_laptrack_object(max_distance):
    lt = LapTrack(
        metric="sqeuclidean",  # The similarity metric for particles. See `scipy.spatial.distance.cdist` for allowed values.
        splitting_metric="sqeuclidean",
        merging_metric="sqeuclidean",
        # the square of the cutoff distance for the "sqeuclidean" metric
        cutoff=max_distance**2,
        splitting_cutoff=False,  # or max_distance**2 for non-splitting case
        merging_cutoff=False,  # or max_distance**2 for merging case
    )
    return lt

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    
def run_laptrack(lt, spots_df):
    track_df, split_df, merge_df = lt.predict_dataframe(
        spots_df.assign(weighted_cell_id=spots_df["cell_id"] * 100),
        coordinate_cols=[
            "position_x",
            "position_y",
            "weighted_cell_id",
        ],  # the column names for the coordinates
        frame_col="frame",  # the column name for the frame (default "frame")
    )

    return track_df

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def map_locus_to_cell(track_df, local_cell_registry):
    """
    Maps an (X, Y) image coordinate to normalized cellular coordinates.
    
    Parameters:
    mesh_grid: numpy array of shape (N, M, 2)
    locus_xy: tuple or list (X, Y) of the fluorescent spot
    
    Returns:
    long_frac: float (0.0 to 1.0) representing distance from Pole A to Pole B
    lat_frac: float (0.0 to 1.0) representing distance from Left Wall to Right Wall
    """
    normalized_coords = []
    for idx, spot in track_df.iterrows():
        # Safety check: If it's a background spot (-1) or missing cell, return NaNs
        if spot.cell_id not in local_cell_registry or spot.cell_id == -1:
            output.append((np.nan, np.nan))
            continue
            
        # Extract the specific cell object
        cell = local_cell_registry[spot.cell_id]
        mesh_grid = cell.mesh_grid
        
        # Translate global locus to local mesh coordinates
        local_x = spot.position_x - cell.x_offset
        local_y = spot.position_y - cell.y_offset
        locus_xy = (local_x, local_y)
        # 1. Calculate Euclidean distance from the locus to EVERY node in the mesh
        # np.linalg.norm with axis=2 calculates the distance across the (X, Y) pairs
        distances = np.linalg.norm(mesh_grid - locus_xy, axis=2)
    
        # 2. Find the 2D index (i, j) of the node with the smallest distance
        # np.argmin gives a flat index, unravel_index converts it back to (i, j)
        closest_i, closest_j = np.unravel_index(np.argmin(distances), distances.shape)
        
        # 3. Convert array indices into biological fractions
        N, M, _ = mesh_grid.shape
        
        longitudinal_fraction = closest_i / (N - 1)
        lateral_fraction = (closest_j / (M - 1)) - 0.5
        normalized_coords.append((lateral_fraction, longitudinal_fraction))

    normalized_coords = np.vstack(normalized_coords)
    updated_df = track_df.copy()
    updated_df["cellular_position_x"] = normalized_coords[:,0]
    updated_df["cellular_position_y"] = normalized_coords[:,1]
    return updated_df

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def enforce_track_length_minimum(tracks_dataframe, minimum_track_length):
    
    track_lengths = tracks_dataframe.groupby('track_id')['track_id'].transform('size')
    filtered_tracks_df = tracks_dataframe[track_lengths >= minimum_track_length].copy()
    
    return filtered_tracks_df

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------