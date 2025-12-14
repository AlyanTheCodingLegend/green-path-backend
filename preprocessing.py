"""
Preprocessing module for GreenPath.
Aggregates satellite data to hexagon grid and calculates derived metrics.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import pickle
import os

from config import CACHE_DIR, BBOX


def aggregate_to_hexagons(hex_grid, point_data, value_column, method='mean'):
    """
    Aggregate point data to hexagon grid using spatial join.

    Args:
        hex_grid: GeoDataFrame with hexagon geometries
        point_data: DataFrame with lat, lon, and value columns
        value_column: Name of the column to aggregate
        method: Aggregation method ('mean', 'median', 'max', 'min')

    Returns:
        Series with aggregated values indexed by hex_id
    """
    # Convert points to GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        point_data.copy(),
        geometry=gpd.points_from_xy(point_data['lon'], point_data['lat']),
        crs='EPSG:4326'
    )

    # Ensure hex_grid has hex_id as column
    hex_for_join = hex_grid.copy()
    if 'hex_id' not in hex_for_join.columns:
        hex_for_join = hex_for_join.reset_index()

    # Spatial join - join hexagons to points
    joined = gpd.sjoin(points_gdf, hex_for_join[['hex_id', 'geometry']],
                       how='inner', predicate='within')

    if len(joined) == 0:
        # Fallback: use nearest hexagon for each point
        print(f"  Warning: No points within hexagons, using nearest neighbor...")
        from scipy.spatial import cKDTree

        # Get hexagon centroids
        hex_centroids = np.array([(geom.centroid.x, geom.centroid.y)
                                   for geom in hex_for_join.geometry])
        point_coords = np.array(list(zip(point_data['lon'], point_data['lat'])))

        # Find nearest hexagon for each point
        tree = cKDTree(hex_centroids)
        _, indices = tree.query(point_coords)

        # Assign hex_id to points
        point_data_with_hex = point_data.copy()
        point_data_with_hex['hex_id'] = hex_for_join.iloc[indices]['hex_id'].values

        # Aggregate
        if method == 'mean':
            aggregated = point_data_with_hex.groupby('hex_id')[value_column].mean()
        elif method == 'median':
            aggregated = point_data_with_hex.groupby('hex_id')[value_column].median()
        else:
            aggregated = point_data_with_hex.groupby('hex_id')[value_column].mean()

        return aggregated

    # Aggregate by hexagon
    if method == 'mean':
        aggregated = joined.groupby('hex_id')[value_column].mean()
    elif method == 'median':
        aggregated = joined.groupby('hex_id')[value_column].median()
    elif method == 'max':
        aggregated = joined.groupby('hex_id')[value_column].max()
    elif method == 'min':
        aggregated = joined.groupby('hex_id')[value_column].min()
    else:
        aggregated = joined.groupby('hex_id')[value_column].mean()

    return aggregated


def interpolate_missing_values(hex_grid, values_series, value_name):
    """
    Interpolate missing values using IDW (Inverse Distance Weighting).

    Args:
        hex_grid: GeoDataFrame with hexagon geometries
        values_series: Series with values indexed by hex_id
        value_name: Name for the interpolated column

    Returns:
        Series with interpolated values for all hexagons
    """
    # Merge values with grid
    hex_with_values = hex_grid.copy()

    # Ensure hex_id is a column, not index
    if 'hex_id' not in hex_with_values.columns:
        hex_with_values = hex_with_values.reset_index()

    # Map values from series to hexagons
    hex_with_values[value_name] = hex_with_values['hex_id'].map(values_series)

    # Separate known and unknown
    known = hex_with_values[hex_with_values[value_name].notna()]
    unknown = hex_with_values[hex_with_values[value_name].isna()]

    if len(unknown) == 0:
        return hex_with_values.set_index('hex_id')[value_name]

    if len(known) == 0:
        # No data available, return default based on value type
        if 'ndvi' in value_name.lower():
            default_val = 0.3
        elif 'lst' in value_name.lower():
            default_val = 35.0
        elif 'slope' in value_name.lower():
            default_val = 2.0
        else:
            default_val = 0.5
        hex_with_values[value_name] = default_val
        return hex_with_values.set_index('hex_id')[value_name]

    # Get coordinates
    known_coords = np.array([(g.centroid.x, g.centroid.y) for g in known.geometry])
    unknown_coords = np.array([(g.centroid.x, g.centroid.y) for g in unknown.geometry])
    known_values = known[value_name].values

    # Interpolate using griddata
    interpolated = griddata(
        known_coords, known_values, unknown_coords,
        method='linear', fill_value=np.nanmean(known_values)
    )

    # Update missing values
    hex_with_values.loc[hex_with_values[value_name].isna(), value_name] = interpolated

    return hex_with_values.set_index('hex_id')[value_name]


def calculate_building_shadow_index(hex_grid, buildings):
    """
    Calculate building shadow potential for each hexagon.
    Higher values indicate more potential shade from buildings.

    Args:
        hex_grid: GeoDataFrame with hexagon geometries
        buildings: GeoDataFrame with building footprints and heights

    Returns:
        Series with shadow index (0-1) indexed by hex_id
    """
    if buildings is None or len(buildings) == 0:
        # Return default low shadow values
        return pd.Series(0.1, index=hex_grid['hex_id'])

    shadow_index = {}

    for idx, hex_row in hex_grid.iterrows():
        hex_geom = hex_row.geometry
        hex_area = hex_geom.area

        # Find buildings that intersect with this hexagon
        intersecting = buildings[buildings.intersects(hex_geom)]

        if len(intersecting) == 0:
            shadow_index[hex_row['hex_id']] = 0
            continue

        # Calculate shadow potential based on building coverage and height
        total_shadow = 0
        for _, building in intersecting.iterrows():
            intersection = building.geometry.intersection(hex_geom)
            if intersection.is_empty:
                continue

            coverage = intersection.area / hex_area
            height = building.get('height_m', 6)

            # Shadow potential increases with height (max effect at 15m+)
            height_factor = min(height / 15, 1)

            total_shadow += coverage * height_factor

        # Normalize to 0-1 range
        shadow_index[hex_row['hex_id']] = min(total_shadow, 1)

    return pd.Series(shadow_index)


def normalize_values(series, invert=False, global_min=None, global_max=None):
    """
    Normalize values to 0-1 range using global or local min-max scaling.

    Args:
        series: Pandas Series to normalize
        invert: If True, invert so that low values become high
        global_min: Fixed minimum value for normalization (for cross-city consistency)
        global_max: Fixed maximum value for normalization (for cross-city consistency)

    Returns:
        Normalized Series
    """
    # Use global ranges if provided, otherwise use data min/max
    if global_min is not None and global_max is not None:
        min_val = global_min
        max_val = global_max
    else:
        min_val = series.min()
        max_val = series.max()

    if max_val == min_val:
        return pd.Series(0.5, index=series.index)

    normalized = (series - min_val) / (max_val - min_val)

    # Clip values to 0-1 range (handles values outside global range)
    normalized = normalized.clip(0, 1)

    if invert:
        normalized = 1 - normalized

    return normalized


def preprocess_data(data):
    """
    Main preprocessing function.
    Aggregates all data to hexagon grid and calculates normalized scores.
    Uses global normalization ranges for cross-city consistency.

    Args:
        data: Dictionary with raw data from data collection

    Returns:
        GeoDataFrame with hexagon grid and all preprocessed features
    """
    # Global normalization ranges for Pakistani cities
    # These ensure all cities are scored on the same scale
    GLOBAL_RANGES = {
        'ndvi': (-0.3, 0.8),    # Typical NDVI range for urban areas
        'lst': (25.0, 55.0),     # Typical LST range in Celsius for summer
        'slope': (0.0, 15.0),    # Walkable slope range in degrees
        'shadow': (0.0, 1.0)     # Already normalized
    }

    # Note: Caching is handled at the app level per-city
    # Don't use internal caching here to avoid cross-city cache conflicts

    print("\n" + "="*50)
    print("Preprocessing Data")
    print("="*50 + "\n")

    hex_grid = data['hex_grid'].copy()

    # Set hex_id as index for easier merging
    if 'hex_id' in hex_grid.columns:
        hex_grid = hex_grid.set_index('hex_id')

    # 1. Aggregate NDVI to hexagons
    print("Aggregating NDVI data to hexagons...")
    ndvi_agg = aggregate_to_hexagons(data['hex_grid'], data['ndvi'], 'ndvi')
    ndvi_interpolated = interpolate_missing_values(data['hex_grid'], ndvi_agg, 'ndvi')
    hex_grid['ndvi'] = ndvi_interpolated
    hex_grid['ndvi_score'] = normalize_values(
        hex_grid['ndvi'],
        global_min=GLOBAL_RANGES['ndvi'][0],
        global_max=GLOBAL_RANGES['ndvi'][1]
    )
    print(f"  NDVI range: {hex_grid['ndvi'].min():.3f} - {hex_grid['ndvi'].max():.3f}")
    print(f"  NDVI score range: {hex_grid['ndvi_score'].min():.3f} - {hex_grid['ndvi_score'].max():.3f}")

    # 2. Aggregate LST to hexagons
    print("Aggregating LST data to hexagons...")
    lst_agg = aggregate_to_hexagons(data['hex_grid'], data['lst'], 'lst')
    lst_interpolated = interpolate_missing_values(data['hex_grid'], lst_agg, 'lst')
    hex_grid['lst'] = lst_interpolated
    # Invert LST so that cooler = better
    hex_grid['lst_score'] = normalize_values(
        hex_grid['lst'],
        invert=True,
        global_min=GLOBAL_RANGES['lst'][0],
        global_max=GLOBAL_RANGES['lst'][1]
    )
    print(f"  LST range: {hex_grid['lst'].min():.1f}C - {hex_grid['lst'].max():.1f}C")
    print(f"  LST score range: {hex_grid['lst_score'].min():.3f} - {hex_grid['lst_score'].max():.3f}")

    # 3. Aggregate slope to hexagons
    print("Aggregating slope data to hexagons...")
    slope_agg = aggregate_to_hexagons(data['hex_grid'], data['slope'], 'slope')
    slope_interpolated = interpolate_missing_values(data['hex_grid'], slope_agg, 'slope')
    hex_grid['slope'] = slope_interpolated
    # Invert slope so that flatter = better
    hex_grid['slope_score'] = normalize_values(
        hex_grid['slope'],
        invert=True,
        global_min=GLOBAL_RANGES['slope'][0],
        global_max=GLOBAL_RANGES['slope'][1]
    )
    print(f"  Slope range: {hex_grid['slope'].min():.1f} - {hex_grid['slope'].max():.1f}")
    print(f"  Slope score range: {hex_grid['slope_score'].min():.3f} - {hex_grid['slope_score'].max():.3f}")

    # 4. Calculate building shadow index
    print("Calculating building shadow potential...")
    shadow_series = calculate_building_shadow_index(data['hex_grid'], data['buildings'])
    hex_grid['shadow'] = shadow_series
    hex_grid['shadow_score'] = normalize_values(
        hex_grid['shadow'],
        global_min=GLOBAL_RANGES['shadow'][0],
        global_max=GLOBAL_RANGES['shadow'][1]
    )
    print(f"  Shadow coverage: {(hex_grid['shadow'] > 0).sum()} hexagons with buildings")

    print("\n[OK] Preprocessing complete!")
    print(f"  Total hexagons: {len(hex_grid)}")

    return hex_grid


def get_edge_hexagons(edge_geometry, hex_grid):
    """
    Get all hexagons that a road edge passes through.

    Args:
        edge_geometry: LineString geometry of the edge
        hex_grid: GeoDataFrame with hexagon geometries (with hex_id as index)

    Returns:
        List of hex_ids that the edge intersects
    """
    intersecting = hex_grid[hex_grid.intersects(edge_geometry)]
    return list(intersecting.index)


if __name__ == '__main__':
    # Test preprocessing
    from data_collection import collect_all_data

    data = collect_all_data()
    hex_grid = preprocess_data(data)

    print("\nPreprocessed data summary:")
    print(hex_grid[['ndvi', 'lst', 'slope', 'shadow',
                    'ndvi_score', 'lst_score', 'slope_score', 'shadow_score']].describe())
