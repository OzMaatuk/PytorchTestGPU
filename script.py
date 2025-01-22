
# %%
import torch

# %%
# Check if a GPU is available
torch.cuda.is_available()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the GPU name and memory
if device.type == "cuda":
    print("GPU name:", torch.cuda.get_device_name(device))
    print("GPU memory:", torch.cuda.get_device_properties(device).total_memory)
else:
    print("No GPU available")

# Create a random tensor on the GPU
tensor = torch.randn((10000, 1000), device=device)

# Compute the mean of the tensor
mean = tensor.mean()

# Print the mean
print("Mean:", mean)

# %%
import numpy as np
import time

# Create a large random array
array = np.random.randn(10000000)

# Copy the array to the GPU
gpu_array = torch.tensor(array, device="cuda")

# Start the timer
start = time.time()

# Compute the sum of the elements of the array on the GPU
sum = gpu_array.sum()

# Stop the timer
end = time.time()

# Print the execution time
print("Execution time:", end - start)

# %%



