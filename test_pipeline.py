"""Test script for GreenPath pipeline."""
import sys
sys.path.insert(0, '.')

from data_collection import collect_all_data
from preprocessing import preprocess_data
from scoring import calculate_comfort_scores

print('Testing GreenPath data pipeline...')
print()

# Collect data
data = collect_all_data()

print()
print('Hex grid columns:', list(data['hex_grid'].columns))
print('Hex grid index:', data['hex_grid'].index.name)
print('NDVI data shape:', data['ndvi'].shape)
print('NDVI sample:')
print(data['ndvi'].head(3))
print()

# Preprocess
hex_grid = preprocess_data(data)

print()
print('After preprocessing:')
print('Columns:', list(hex_grid.columns))
print('Index name:', hex_grid.index.name)
print()
print('NDVI values:')
print(hex_grid['ndvi'].describe())
print()
print('LST values:')
print(hex_grid['lst'].describe())
print()

# Score
scored = calculate_comfort_scores(hex_grid)
print()
print('Comfort scores:')
print(scored['comfort_score'].describe())
