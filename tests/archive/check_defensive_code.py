#!/usr/bin/env python
"""
Check if defensive NaN/Inf handling is present in distributions.py
"""

import inspect
from bayesDREAM.fitting.distributions import sample_binomial_trans

# Get source code
source = inspect.getsource(sample_binomial_trans)

# Check for defensive code markers
has_defensive_check = "# Defensive: Check for NaN/Inf before passing to pyro.factor" in source
has_replacement = "torch.where(torch.isfinite(log_prob)" in source
has_warning = "[WARNING] Replacing" in source and "non-finite log_prob values" in source

print("=" * 60)
print("Checking for defensive NaN/Inf handling in distributions.py")
print("=" * 60)
print()

if has_defensive_check and has_replacement and has_warning:
    print("✓ PASS: Defensive code is present")
    print()
    print("Your distributions.py has the updated code that handles")
    print("non-finite values during Predictive sampling.")
else:
    print("✗ FAIL: Defensive code is MISSING")
    print()
    print("Your distributions.py is running an old version without")
    print("NaN/Inf handling. You need to update the file.")
    print()
    print("Markers found:")
    print(f"  - Defensive comment: {has_defensive_check}")
    print(f"  - torch.isfinite check: {has_replacement}")
    print(f"  - Warning message: {has_warning}")

print()
print("Source location:")
print(f"  {inspect.getfile(sample_binomial_trans)}")
print()
