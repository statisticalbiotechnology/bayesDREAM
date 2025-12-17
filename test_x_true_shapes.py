import torch

# Simulate point estimate case
print("="*60)
print("POINT ESTIMATE CASE")
print("="*60)

x_true_point = torch.randn(10)  # [N=10]
x_true_type = 'point'
samp = 0

print(f"x_true shape: {x_true_point.shape}")
print(f"x_true_type: {x_true_type}")
print(f"samp: {samp}")
print(f"x_true.shape[0]: {x_true_point.shape[0]}")
print(f"samp < x_true.shape[0]: {samp < x_true_point.shape[0]}")

# Simulate the ternary from line 1732-1735
x_true_sample = (
    x_true_point[samp] if samp < x_true_point.shape[0] else x_true_point.mean(dim=0)
    if x_true_type == "posterior" else x_true_point
)

print(f"x_true_sample shape: {x_true_sample.shape}")
print(f"x_true_sample value (first 5): {x_true_sample[:5]}")
print(f"x_true value (first 5): {x_true_point[:5]}")
print(f"Are they the same? {torch.allclose(x_true_sample, x_true_point)}")

# Simulate posterior case  
print("\n" + "="*60)
print("POSTERIOR CASE")
print("="*60)

x_true_posterior = torch.randn(100, 10)  # [S=100, N=10]
x_true_type_post = 'posterior'
samp_post = 42

print(f"x_true shape: {x_true_posterior.shape}")
print(f"x_true_type: {x_true_type_post}")
print(f"samp: {samp_post}")

x_true_sample_post = (
    x_true_posterior[samp_post] if samp_post < x_true_posterior.shape[0] else x_true_posterior.mean(dim=0)
    if x_true_type_post == "posterior" else x_true_posterior
)

print(f"x_true_sample shape: {x_true_sample_post.shape}")
print(f"x_true_sample is one posterior sample? {torch.allclose(x_true_sample_post, x_true_posterior[samp_post])}")

