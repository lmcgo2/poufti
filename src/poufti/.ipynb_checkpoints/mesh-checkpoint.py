import math
import numpy as np
import scipy
import shapely

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_max_indices(condensed_array, m):
    # get the 1D index of the maximum value
    k = np.argmax(condensed_array)
    
    # calculate the row index (i)
    b = 2 * m - 1
    i = math.floor((b - math.sqrt(b**2 - 8 * k)) / 2)
    
    # calculate the column index (j)
    j = k - m * i + ((i + 2) * (i + 1)) // 2
    
    return int(i), int(j)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def split_boundary(cell, num_longitudinal_points):

    if num_longitudinal_points % 2 != 1:
        num_longitudinal_points +=1
        print(f'split_boundary: num_longitudinal_points cannot be even; correcting to num_lateral_points = {M}')

    poly_coords = np.vstack(cell.local_polygon.exterior.coords[:])
    distance_matrix = scipy.spatial.distance.pdist(poly_coords)
    
    i,j = get_max_indices(distance_matrix, len(poly_coords))
    
    pole_a, pole_b = poly_coords[i], poly_coords[j]
    x,y = cell.local_polygon.exterior.xy
    right_wall_coords = np.stack((x[i:j], y[i:j]), axis=1)
    left_wall_coords = np.stack((np.concatenate((x[j:],x[0:i])), np.concatenate((y[j:],y[0:i]))), axis=1)
    
    right_wall = shapely.geometry.LineString(right_wall_coords)
    left_wall = shapely.geometry.LineString(left_wall_coords)

    xs = np.linspace(0, 1, num_longitudinal_points)
    
    right_wall_points = np.vstack(shapely.get_coordinates(right_wall.interpolate(xs, normalized=True)))
    left_wall_points = np.vstack(shapely.get_coordinates(left_wall.interpolate(xs, normalized=True)))

    left_wall_points = left_wall_points[::-1]

    right_wall_points[0] = pole_a
    left_wall_points[0] = pole_a
    
    right_wall_points[-1] = pole_b
    left_wall_points[-1] = pole_b

    return left_wall_points, right_wall_points

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_internal_mesh(left_wall_points, right_wall_points, num_lateral_points):
    """
    Constructs a perfectly structured (N, M, 2) mesh grid from aligned cell walls.
    """
    M = num_lateral_points
    if M % 2 != 1:
        M +=1
        print(f'build_internal_mesh: num_lateral_points cannot be even; correcting to num_lateral_points = {M}')
    # N is naturally dictated by your num_of_points + 2
    N = len(left_wall_points) 
    
    # Initialize the Final Data Cube
    mesh_grid = np.zeros((N, M, 2))
    
    # Iterate straight through the matched indicess
    for k in range(N):
        mesh_grid[k] = np.linspace(left_wall_points[k], right_wall_points[k], M)
        
    return mesh_grid

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def refine_mesh_to_equidistant(mesh_grid, N_points=100, M_points=16):
    """
    Refines an existing uneven mesh grid so that its rows are spaced
    at perfectly equal physical distances along the cell centerline.
    """
    # 1. Get the current centerline and its physical spacing
    current_centerline = mesh_grid.mean(axis=1) # Shape: (N, 2)
    dr = np.diff(current_centerline, axis=0)
    step_lengths = np.linalg.norm(dr, axis=1)
    
    # Calculate the normalized cumulative distance profile (0.0 to 1.0)
    arc_lengths = np.insert(np.cumsum(step_lengths), 0, 0.0)
    u_current = arc_lengths / arc_lengths[-1] # Current row positions in %
    
    # 2. Extract the current raw outer boundaries
    left_wall = mesh_grid[:, 0, :]   # Shape: (N, 2)
    right_wall = mesh_grid[:, -1, :] # Shape: (N, 2)
    
    # 3. Create our target target grid: perfectly uniform steps from 0.0 to 1.0
    u_uniform = np.linspace(0, 1, N_points)
    
    # 4. Interpolate the new wall positions along the uniform grid
    left_wall_new = np.zeros((N_points, 2))
    right_wall_new = np.zeros((N_points, 2))
    
    for dim in range(2): # Map X (0) and Y (1) independently
        left_wall_new[:, dim] = np.interp(u_uniform, u_current, left_wall[:, dim])
        right_wall_new[:, dim] = np.interp(u_uniform, u_current, right_wall[:, dim])
        
    # 5. Reconstruct the internal mesh nodes linearly between the new walls
    refined_mesh = np.zeros((N_points, M_points, 2))
    for i in range(N_points):
        # Linearly space M_points between the left wall and right wall for row i
        refined_mesh[i, :, 0] = np.linspace(left_wall_new[i, 0], right_wall_new[i, 0], M_points)
        refined_mesh[i, :, 1] = np.linspace(left_wall_new[i, 1], right_wall_new[i, 1], M_points)
        
    return refined_mesh

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------