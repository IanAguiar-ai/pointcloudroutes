import pandas as pd
import numpy as np

def discover_route(df:pd.DataFrame, initial_point:list[float, float], meters:int = 5, lon_col:str = "lon", lat_col:str = "lat",
                   angular_samples:int = 360, iterations:int = 1000, increase_meters:float = 1.5,
                   decrease_meters:float = 2/3, max_meters:float = 100, to_maintain:list = [],
                   verbose:bool = False) -> pd.DataFrame:
    """
    Discovers a sequential route through a spatial point cloud based on point density,
    proximity scoring, and angular constraints.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing spatial coordinates and optional metadata.
    initial_point : list[float, float]
        Starting coordinate as a [longitude, latitude] pair.
    meters : int, default 5
        Initial search radius step in meters.
    lon_col : str, default "lon"
        Name of the longitude column in `df`.
    lat_col : str, default "lat"
        Name of the latitude column in `df`.
    angular_samples : int, default 64
        Number of candidate points to generate circularly around the current step.
    iterations : int, default 300
        Maximum number of routing steps to perform.
    increase_meters : float, default 1.5
        Multiplier to expand the search radius when no points are found.
    decrease_meters : float, default 0.67
        Multiplier to shrink the search radius back down after a successful step.
    max_meters : float, default 100
        Maximum allowable search radius in meters before stopping.
    to_maintain : list, default []
        Columns from `df` to aggregate (by mean) and append to each route point.

    Returns:
    --------
    pd.DataFrame
        Generated route containing coordinates and aggregated variables.
    """

    # 1. Filter dataset to include only tracking coordinates and metadata to track
    cols_to_keep = [lon_col, lat_col] + (to_maintain if to_maintain else [])
    df = df[cols_to_keep].copy()

    # Pre-compute spatial delta vectors (direction vectors) for every point in the dataset
    df_diff_lon = df[lon_col].diff().to_numpy()
    df_diff_lat = df[lat_col].diff().to_numpy()

    # Convert core data structures to raw NumPy arrays for high-performance vectorized operations
    df_indices = df.index.to_numpy()
    df_lons = df[lon_col].to_numpy()
    df_lats = df[lat_col].to_numpy()
    df_maintain = df[to_maintain].to_numpy() if to_maintain else None

    # Use a boolean mask to track active points instead of expensive pd.DataFrame.drop operations
    # First row is disabled (False) because it yields NaN values during the .diff() calculation
    active_mask = np.ones(len(df), dtype=bool)
    active_mask[0] = False

    # Convert threshold and working distances from meters to approximate degrees/radians
    meters_rad = meters / 111_132
    meters_min_rad = meters_rad
    max_meters_rad = max_meters * 111_132

    n = 0
    point_now = np.array(initial_point, dtype=float)

    # Initialize lists to gather route results efficiently
    route_lons, route_lats = [], []
    dist_meters = []
    route_maintain = {col: [] for col in to_maintain} if to_maintain else {}
    points_captured = []

    # Pre-calculate trigonometric components for candidate point generation
    angles = np.linspace(0, 2 * np.pi, angular_samples, endpoint=False)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    # Main iterative routing loop
    while (n < iterations) and (meters_rad <= max_meters_rad):
        if verbose:
            print(f"\r{n} | ({meters_rad * 111_132:04.02f})", end="")
        n += 1

        # Generate candidate coordinates dynamically around the current position
        possible_points_lon = point_now[0] + meters_rad * cos_angles
        possible_points_lat = point_now[1] + meters_rad * sin_angles

        # Isolate indices of remaining uncaptured points
        active_idx = np.where(active_mask)[0]
        if len(active_idx) == 0:
            break

        lons_act = df_lons[active_idx]
        lats_act = df_lats[active_idx]
        diff_lons_act = df_diff_lon[active_idx]
        diff_lats_act = df_diff_lat[active_idx]

        # Calculate distances from the current route point (p1) to all active dataset points
        dist1 = np.sqrt((lons_act - point_now[0])**2 + (lats_act - point_now[1])**2)
        mask_r1 = dist1 <= meters_rad

        # Vectorized Directional/Angular Filtering (Replicating original cosine curve logic)
        # Only enforced once the route contains at least two points to establish a heading vector
        if len(route_lons) > 1:
            # Extract heading vector from the last step of the route
            v_route_lon = route_lons[-1] - route_lons[-2]
            v_route_lat = route_lats[-1] - route_lats[-2]
            norm_route = np.sqrt(v_route_lon**2 + v_route_lat**2)

            if norm_route > 0:
                # Dot product between current route heading and dataset point delta vectors
                dot_product = (diff_lons_act * v_route_lon) + (diff_lats_act * v_route_lat)
                norm_pts = np.sqrt(diff_lons_act**2 + diff_lats_act**2)

                # Safeguard against division by zero for stationary points
                norm_pts[norm_pts == 0] = 1e-9

                # Compute directional cosine similarity
                cos_theta = dot_product / (norm_route * norm_pts)

                # Filter to only allow smooth curves (heading mismatch angle < 90 degrees)
                curve_mask = cos_theta > 0
            else:
                curve_mask = np.ones(len(active_idx), dtype=bool)
        else:
            curve_mask = np.ones(len(active_idx), dtype=bool)

        best_score = -1.0
        best_point = None
        best_captured_idx = np.array([], dtype=int)

        # Evaluate proximity score for each generated angular candidate point
        for i in range(angular_samples):
            p2_lon = possible_points_lon[i]
            p2_lat = possible_points_lat[i]

            # Calculate distance from candidate point (p2) to all active dataset points
            dist2 = np.sqrt((lons_act - p2_lon)**2 + (lats_act - p2_lat)**2)
            mask_r2 = dist2 <= meters_rad

            # Combine radius constraints and the directional/heading curve mask
            combined_mask = (mask_r1 | mask_r2) & curve_mask

            if not np.any(combined_mask):
                continue

            # Inverse-distance density scoring formula
            scores = (1.0 / dist1[combined_mask]) * (1.0 / dist2[combined_mask])
            total_score = np.sum(scores)

            # Keep track of the highest scoring candidate point
            if total_score > best_score:
                best_score = total_score
                best_point = (p2_lon, p2_lat)
                best_captured_idx = active_idx[combined_mask]

        # Handle routing steps and adaptive search radius management
        if best_score <= 0:
            # If no dense point clusters found, expand search area
            meters_rad *= increase_meters
        else:
            # Append winning coordinates to route arrays
            route_lons.append(best_point[0])
            route_lats.append(best_point[1])
            dist_meters.append(meters_rad * 111_132)

            # Deactivate captured points to prevent reprocessing them in subsequent steps
            active_mask[best_captured_idx] = False
            points_captured.append(df.iloc[best_captured_idx])

            # Compute mean values of specified target variables from captured points
            if to_maintain and len(best_captured_idx) > 0:
                mean_vals = np.mean(df_maintain[best_captured_idx], axis=0)
                for idx, col in enumerate(to_maintain):
                    route_maintain[col].append(mean_vals[idx])
            elif to_maintain:
                for col in to_maintain:
                    route_maintain[col].append(np.nan)

            # Advance route state and contract radius back towards base size
            point_now = np.array(best_point)
            meters_rad = max(meters_min_rad, meters_rad * decrease_meters)

    # Format collected tracking data back into a standard output DataFrame
    route_data = {lon_col:route_lons, lat_col:route_lats, "_meters_":dist_meters}
    if to_maintain:
        route_data.update(route_maintain)
    route_df = pd.DataFrame(route_data)

    return route_df
