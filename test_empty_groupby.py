"""
Test what groupby().mean().reset_index() returns with zero rows.
"""
import pandas as pd
import numpy as np

print("Testing groupby behavior with empty DataFrames...")

# Scenario 1: Normal case
print("\n=== Scenario 1: Normal groupby ===")
df = pd.DataFrame({
    'lane': ['lane1', 'lane1', 'lane2'],
    'cell_line': ['K562', 'K562', 'HEK293T'],
    'sum_factor': [1.0, 1.2, 0.9]
})
result = df.groupby(['lane', 'cell_line'])['sum_factor'].mean().reset_index(name='mean_SumFacs')
print(f"result:\n{result}")
print(f"Columns: {result.columns.tolist()}")

# Scenario 2: Empty DataFrame with columns
print("\n=== Scenario 2: Groupby on empty DataFrame (but columns exist) ===")
df_empty = pd.DataFrame(columns=['lane', 'cell_line', 'sum_factor'])
df_empty['sum_factor'] = df_empty['sum_factor'].astype(float)  # Ensure numeric type
try:
    result = df_empty.groupby(['lane', 'cell_line'])['sum_factor'].mean().reset_index(name='mean_SumFacs')
    print(f"result:\n{result}")
    print(f"result shape: {result.shape}")
    print(f"Columns: {result.columns.tolist()}")
    print(f"Is empty: {len(result) == 0}")
except Exception as e:
    print(f"✗ Error: {e}")
    print("Empty DataFrames cause errors with groupby().mean()!")

# Scenario 3: DataFrame where filter returns zero rows
print("\n=== Scenario 3: Groupby after filtering to zero rows ===")
df = pd.DataFrame({
    'target': ['GFI1B', 'GFI1B', 'GFI1B'],
    'lane': ['lane1', 'lane1', 'lane2'],
    'cell_line': ['K562', 'K562', 'HEK293T'],
    'sum_factor': [1.0, 1.2, 0.9]
})
df_filtered = df.loc[df['target'] == 'ntc']  # Zero rows
print(f"df_filtered shape: {df_filtered.shape}")
result = df_filtered.groupby(['lane', 'cell_line'])['sum_factor'].mean().reset_index(name='mean_SumFacs_ntc')
print(f"result:\n{result}")
print(f"result shape: {result.shape}")
print(f"Columns: {result.columns.tolist()}")

# Scenario 4: Try to merge with this empty result
print("\n=== Scenario 4: Merge df_guide with empty df_ntc ===")
df_guide = pd.DataFrame({
    'lane': ['lane1', 'lane2'],
    'cell_line': ['K562', 'HEK293T'],
    'guide_used': ['gRNA1_K562', 'gRNA2_HEK293T'],
    'mean_SumFacs_guide': [1.1, 0.95]
})
df_ntc = result  # Empty from above
merged = pd.merge(df_guide, df_ntc, on=['lane', 'cell_line'], how='left')
print(f"merged:\n{merged}")
print(f"Columns: {merged.columns.tolist()}")
print(f"'mean_SumFacs_ntc' in columns: {'mean_SumFacs_ntc' in merged.columns}")

if 'mean_SumFacs_ntc' in merged.columns:
    print("\n=== Scenario 5: Try to create adjustment_factor ===")
    try:
        merged["adjustment_factor"] = merged["mean_SumFacs_ntc"] / merged["mean_SumFacs_guide"]
        print(f"adjustment_factor created successfully")
        print(f"merged:\n{merged}")
    except KeyError as e:
        print(f"✗ KeyError: {e}")
else:
    print("\n=== Scenario 5: mean_SumFacs_ntc column missing! ===")
    print("This is the bug: when df_ntc is empty, the merge doesn't add mean_SumFacs_ntc")
    try:
        merged["adjustment_factor"] = merged["mean_SumFacs_ntc"] / merged["mean_SumFacs_guide"]
    except KeyError as e:
        print(f"✗ KeyError when trying to access mean_SumFacs_ntc: {e}")
