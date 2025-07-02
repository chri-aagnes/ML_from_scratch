import numpy as np
import prefetch

# Create batch: shape (batch_size, feature_size)
batch = np.random.randn(8, 1024).astype(np.float32)

# Call the C++ function
prefetch.process_batch(batch)

print(batch[0, 0])  # Should be >= 0 due to ReLU
