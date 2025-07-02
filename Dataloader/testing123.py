import numpy as np
import prefetch

X = np.random.randn(8, 1024).astype(np.float32)
y = np.random.randint(0, 10, size=(8,), dtype=np.int32)

print("X before processing:")
print(X)

# call function 
prefetch.process_batch(X, y)

print("\nX after processing (should have ReLU applied):")
print(X)

# Check that no value is negative
assert np.all(X >= 0), "ReLU failed: X contains negative values"

print("\nprefetch is working correctly!")
