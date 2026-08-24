import cellpose
import math
# import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import shapely
from poufti.structures import CellData

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_segmentation(image, model, diameter, cellprob_threshold, flow_threshold, min_size, remove_edge_masks):
    masks, flows, styles = model.eval(
        image, 
        channels=[0, 0],        # greyscale
        diameter=diameter,           # average width of cells. 0 = let cellpose figure out what diameter is best. I tried measuring the diameter but I found
                                # using 0 is better
        cellprob_threshold=cellprob_threshold,  # Decrease if cells are not being picked up, increase if model is picking up noise. [-6.0,6.0]
        flow_threshold=flow_threshold,      # Lower value makes cell outlines more strict and reduces cell merging, increasing allows for more variable shapes [0,1.0]
        min_size=min_size,             # Filters out small debris
        resample=True            # Smoother outlines
    )
    if remove_edge_masks:
        masks = cellpose.utils.remove_edge_masks(masks, change_index=True)
        
    masks = masks.astype(np.uint32)
    
    return masks

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def create_cell_registry(phase_id, masks):
    local_registry = {}
    
    # Get tight bounding box slices for all cells
    cell_slices = scipy.ndimage.find_objects(masks)
    
    # Loop through slices (start=1 ensures cell_id matches mask label)
    for cell_id, cell_slice in enumerate(cell_slices, start=1):
        if cell_slice is None:
            cell = CellData(
                phase_id=phase_id,
                fail_reason="create_cell_registry: cell slicing error"
            )
            continue
            
        cell_slice = tuple(
            slice(max(0, s.start - 3), min(dim_size, s.stop + 3))
            for s, dim_size in zip(cell_slice, masks.shape)
        )
            
        # Grab the localized crop 
        local_crop = masks[cell_slice]
        
        # Find coordinates inside this tiny box
        y_idx, x_idx = np.where(local_crop == cell_id)
        
        # Safety check to skip empty or noisy slices
        if len(y_idx) == 0:
            cell = CellData(
                phase_id=phase_id,
                fail_reason="create_cell_registry: cell slice empty or noisy"
            )
            continue
            
        mask_coords = np.column_stack((y_idx, x_idx))

        # Build cell container
        cell = CellData(
            phase_id=phase_id,
            cell_id=cell_id,
            global_dim = masks.shape,
            local_dim=local_crop.shape,
            local_mask_coords=mask_coords,
            slice_tuple=cell_slice
        )

        local_registry[cell_id] = cell
        
    return local_registry

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def create_polygon(cell, smoothing_proportionality_factor, num_of_intervals):

    outline = cellpose.utils.outlines_list_single(cell.local_binary_mask)[0]

    # do not make polygon if too small
    if len(outline) < 10:
        return None

    # append first position to end so array loops
    initial_pos = outline[0:1, :]
    outline = np.vstack([outline, initial_pos])
    outline = np.ascontiguousarray(outline)

    N = len(outline)
    s = smoothing_proportionality_factor * N

    # tck,u = scipy.interpolate.make_splprep(outline.T, s=s, bc_type='periodic')
    tck, u = scipy.interpolate.splprep(outline.T, s=s, per=1)

    xs = np.linspace(0,1,num_of_intervals)

    # smoothed_outline = tck(xs).T
    smoothed_outline = np.array(scipy.interpolate.splev(xs, tck)).T

    # create polygon
    poly = shapely.geometry.Polygon(smoothed_outline)
    
    # if polygon has boundary line that intersects itself this will recalculate the boundary to fix it
    if not poly.is_valid:
        poly = poly.buffer(0)

    return poly

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------