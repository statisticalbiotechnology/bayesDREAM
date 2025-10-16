"""
Test what happens if df_guide or df_ntc is empty.
"""
import pandas as pd
import numpy as np

print("Testing pandas merge behavior with empty DataFrames...")

# Scenario 1: Normal case
print("\n=== Scenario 1: Normal merge ===")
df1 = pd.DataFrame({'key1': ['A', 'B'], 'key2': [1, 2], 'val1': [10, 20]})
df2 = pd.DataFrame({'key1': ['A', 'B'], 'key2': [1, 2], 'val2': [100, 200]})
merged = pd.merge(df1, df2, on=['key1', 'key2'], how='left')
print(f"merged:\n{merged}")
print(f"Columns: {merged.columns.tolist()}")

# Scenario 2: Empty df2
print("\n=== Scenario 2: Merge with empty df2 ===")
df1 = pd.DataFrame({'key1': ['A', 'B'], 'key2': [1, 2], 'val1': [10, 20]})
df2 = pd.DataFrame(columns=['key1', 'key2', 'val2'])  # Empty but has columns
merged = pd.merge(df1, df2, on=['key1', 'key2'], how='left')
print(f"merged:\n{merged}")
print(f"Columns: {merged.columns.tolist()}")
print(f"'val2' in columns: {'val2' in merged.columns}")

# Scenario 3: Selecting columns from empty DataFrame
print("\n=== Scenario 3: Selecting columns from empty DataFrame ===")
df_empty = pd.DataFrame(columns=['key1', 'key2', 'val2'])
try:
    subset = df_empty[['key1', 'key2', 'val2']]
    print(f"subset:\n{subset}")
    print(f"Columns: {subset.columns.tolist()}")
    print("✓ Selecting columns from empty DataFrame works!")
except KeyError as e:
    print(f"✗ KeyError: {e}")

# Scenario 4: Merge meta with empty subset
print("\n=== Scenario 4: Merge with empty subset from merged DataFrame ===")
meta_out = pd.DataFrame({'key1': ['A', 'B'], 'key2': [1, 2], 'val1': [10, 20]})
merged = pd.DataFrame(columns=['key1', 'key2', 'adjustment_factor'])  # Empty
merge_cols = ['key1', 'key2', 'adjustment_factor']
result = pd.merge(meta_out, merged[merge_cols], on=['key1', 'key2'], how='left')
print(f"result:\n{result}")
print(f"Columns: {result.columns.tolist()}")
print(f"'adjustment_factor' in columns: {'adjustment_factor' in result.columns}")

# Scenario 5: Accessing column from result
print("\n=== Scenario 5: Accessing column from merge result ===")
try:
    result['val1'] * result['adjustment_factor']
    print("✓ Multiplication works (produces NaN)")
except KeyError as e:
    print(f"✗ KeyError: {e}")

# Scenario 6: What if merged doesn't have adjustment_factor at all?
print("\n=== Scenario 6: merged missing adjustment_factor column ===")
meta_out = pd.DataFrame({'key1': ['A', 'B'], 'key2': [1, 2], 'val1': [10, 20]})
merged = pd.DataFrame({'key1': ['A'], 'key2': [1], 'other_col': [100]})  # Missing adjustment_factor!
merge_cols = ['key1', 'key2', 'adjustment_factor']
try:
    result = pd.merge(meta_out, merged[merge_cols], on=['key1', 'key2'], how='left')
    print(f"result:\n{result}")
except KeyError as e:
    print(f"✗ KeyError when trying to select non-existent column: {e}")
    print("THIS is the bug!")
