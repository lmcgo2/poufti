from dataclasses import dataclass
import numpy as np
import shapely
from pathlib import Path
import pandas as pd

@dataclass
class CellData:
    # required identifiers
    phase_id: int
    cell_id: int

    # dimension of image in pixels
    global_dim: tuple = None

    # cropped image
    slice_tuple: tuple = None
    local_dim: tuple = None

    # pixel resolution attributes
    local_mask_coords: np.ndarray = None

    # sub pixel resolution attributes
    local_polygon: shapely.geometry.Polygon = None
    mesh_grid: np.ndarray = None

    # brief reason for invalidness
    fail_reason: str | None = None

    @property
    def y_offset(self) -> int:
        return self.slice_tuple[0].start if self.slice_tuple else 0

    @property
    def x_offset(self) -> int:
        return self.slice_tuple[1].start if self.slice_tuple else 0

    @property
    def is_valid(self) -> bool:
        return self.fail_reason is None

    @property
    def local_binary_mask(self) -> np.ndarray:
        """Fast, tiny mask for skeletonization and morphological math."""
        mask = np.zeros(self.local_dim, dtype=bool)

        if self.local_mask_coords is not None and len(self.local_mask_coords) > 0:
            mask[tuple(self.local_mask_coords.T)] = True
        return mask

    @property
    def global_binary_mask(self) -> np.ndarray:
        """Full-frame mask for exporting or global image math."""
        mask = np.zeros(self.global_dim, dtype=bool)

        if self.local_mask_coords is None or len(self.local_mask_coords) == 0:
            return mask
            
        global_y = self.local_mask_coords[:, 0] + self.y_offset
        global_x = self.local_mask_coords[:, 1] + self.x_offset
        mask[global_y, global_x] = True
        return mask

    @property
    def global_mask_coords(self) -> np.ndarray | None:
        """Convenient pulling of global mask coords"""
        if self.local_mask_coords is None:
            return None
        global_y = self.local_mask_coords[:, 0] + self.y_offset
        global_x = self.local_mask_coords[:, 1] + self.x_offset
        return np.column_stack((global_y, global_x))

    @property
    def global_polygon(self) -> shapely.geometry.Polygon | None:
        """Convenient pulling of global polygon"""
        if self.local_polygon is None:
            return None
        return shapely.affinity.translate(self.local_polygon, xoff=self.x_offset, yoff=self.y_offset)

    @property
    def length(self) -> float:
        """
        Dynamically calculates the midline length from an (N, M, 2) mesh grid.
        """
        if self.mesh_grid is None or self.mesh_grid.shape[0] < 2:
            return 0.0
            
        # 1. Find the exact middle index along the lateral axis (M)
        M = self.mesh_grid.shape[1]
        mid_idx = M // 2
        
        # 2. Slice the entire midline directly out of the 3D cube
        # This gives us an (N, 2) array of pure [X, Y] coordinates down the spine
        midpoints = self.mesh_grid[:, mid_idx, :]
        
        # 3. Calculate distances between consecutive midpoints
        diffs = np.diff(midpoints, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        
        # 4. Sum the segments to get the total curved length
        return float(np.sum(segment_lengths))

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class Dataset:
    """A clean container linking a phase image to its tracking frames."""
    dataset_id: str
    phase_image: Path
    tracking_movie: Path

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class ExperimentData:
    cells: pd.DataFrame
    tracks: pd.DataFrame
    metadata: dict

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------