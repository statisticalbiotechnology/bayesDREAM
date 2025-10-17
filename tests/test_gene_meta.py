"""
Test gene metadata handling in bayesDREAM.
"""
import numpy as np
import pandas as pd
from bayesDREAM import bayesDREAM

print("=" * 80)
print("Test 1: Create model WITHOUT gene_meta (should create minimal metadata)")
print("=" * 80)

# Create mock data
meta = pd.DataFrame({
    'cell': [f'cell{i}' for i in range(1, 21)],
    'guide': ['g1', 'g2', 'g3', 'g4', 'g5'] * 4,
    'target': ['GFI1B'] * 10 + ['ntc'] * 10,
    'cell_line': ['A'] * 10 + ['B'] * 10,
    'sum_factor': [1.0] * 20
})

gene_counts = pd.DataFrame(
    np.random.randint(10, 100, (10, 20)),
    index=[f'GENE{i}' for i in range(10)],
    columns=[f'cell{i}' for i in range(1, 21)]
)
# Make sure GFI1B is in the counts
gene_counts.loc['GFI1B'] = np.random.randint(50, 150, 20)

# Create model without gene_meta
model1 = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    output_dir='./test_output',
    label='test_no_meta'
)

print(f"\n✓ Model created without gene_meta")
print(f"✓ gene_meta shape: {model1.gene_meta.shape}")
print(f"✓ gene_meta columns: {model1.gene_meta.columns.tolist()}")
print(f"✓ First few rows:")
print(model1.gene_meta.head())

print("\n" + "=" * 80)
print("Test 2: Create model WITH gene_meta (full metadata)")
print("=" * 80)

# Create comprehensive gene metadata
gene_meta = pd.DataFrame({
    'gene': [f'GENE{i}' for i in range(10)] + ['GFI1B'],
    'gene_name': [f'GeneSymbol{i}' for i in range(10)] + ['GFI1B_Symbol'],
    'gene_id': [f'ENSG{i:08d}' for i in range(11)],
    'chromosome': ['chr1'] * 11,
    'biotype': ['protein_coding'] * 11
}, index=[f'GENE{i}' for i in range(10)] + ['GFI1B'])

# Create model with gene_meta
model2 = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    gene_meta=gene_meta,
    cis_gene='GFI1B',
    output_dir='./test_output',
    label='test_with_meta'
)

print(f"\n✓ Model created with gene_meta")
print(f"✓ gene_meta shape: {model2.gene_meta.shape}")
print(f"✓ gene_meta columns: {model2.gene_meta.columns.tolist()}")
print(f"✓ First few rows:")
print(model2.gene_meta.head())

print("\n" + "=" * 80)
print("Test 3: gene_meta with only gene_name column")
print("=" * 80)

# Create gene metadata with only gene_name
gene_meta_simple = pd.DataFrame({
    'gene_name': [f'GENE{i}' for i in range(10)] + ['GFI1B']
}, index=[f'GENE{i}' for i in range(10)] + ['GFI1B'])

model3 = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    gene_meta=gene_meta_simple,
    cis_gene='GFI1B',
    output_dir='./test_output',
    label='test_simple_meta'
)

print(f"\n✓ Model created with simple gene_meta (gene_name only)")
print(f"✓ gene_meta shape: {model3.gene_meta.shape}")
print(f"✓ gene_meta columns: {model3.gene_meta.columns.tolist()}")
assert 'gene' in model3.gene_meta.columns, "Should have 'gene' column created"
print("✓ 'gene' column was created from 'gene_name'")

print("\n" + "=" * 80)
print("Test 4: gene_meta index becomes 'gene' column")
print("=" * 80)

# Create gene metadata with named index
gene_meta_indexed = pd.DataFrame({
    'gene_name': [f'GeneSymbol{i}' for i in range(10)] + ['GFI1B_Symbol'],
    'gene_id': [f'ENSG{i:08d}' for i in range(11)]
})
gene_meta_indexed.index = [f'GENE{i}' for i in range(10)] + ['GFI1B']
gene_meta_indexed.index.name = 'gene_symbol'

model4 = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    gene_meta=gene_meta_indexed,
    cis_gene='GFI1B',
    output_dir='./test_output',
    label='test_indexed_meta'
)

print(f"\n✓ Model created with indexed gene_meta")
print(f"✓ gene_meta shape: {model4.gene_meta.shape}")
print(f"✓ gene_meta columns: {model4.gene_meta.columns.tolist()}")
assert 'gene' in model4.gene_meta.columns, "Should have 'gene' column created from index"
print("✓ 'gene' column was created from index")

print("\n" + "=" * 80)
print("ALL GENE METADATA TESTS PASSED!")
print("=" * 80)
print()
print("Summary:")
print("  ✓ Model works without gene_meta (creates minimal metadata)")
print("  ✓ Model accepts comprehensive gene_meta with multiple columns")
print("  ✓ Model handles gene_meta with only gene_name")
print("  ✓ Model uses index as gene identifier when appropriate")
