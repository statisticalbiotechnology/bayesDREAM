import torch

# The exact ternary from the code
x_true = torch.randn(10)  # Point estimate: [N]
x_true_type = 'point'
samp = 0

print("Testing the ternary expression:")
print(f"x_true.shape: {x_true.shape}")
print(f"x_true_type: '{x_true_type}'")
print(f"samp: {samp}")

# The CURRENT code (potentially buggy)
x_true_sample_current = (
    x_true[samp] if samp < x_true.shape[0] else x_true.mean(dim=0)
    if x_true_type == "posterior" else x_true
)

print(f"\nCURRENT CODE:")
print(f"x_true_sample.shape: {x_true_sample_current.shape}")
print(f"Is scalar? {x_true_sample_current.ndim == 0}")

# What it SHOULD be
if x_true_type == "posterior":
    x_true_sample_correct = (
        x_true[samp] if samp < x_true.shape[0] else x_true.mean(dim=0)
    )
else:
    x_true_sample_correct = x_true

print(f"\nCORRECT CODE:")
print(f"x_true_sample.shape: {x_true_sample_correct.shape}")
print(f"Matches original? {torch.equal(x_true_sample_correct, x_true)}")
