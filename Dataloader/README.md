# Dataloader

This module implements the dataloader objects, that will serve as the inputs to our neural networks.

Features that I want it to support:

- Batching: The most important is that it is an iterable interface, such that it is easy to loop through the dataset in batches (of custom size).
- Shuffling: randomizes the order of data samples (to not overfit to sequential patterns in time series)
- Data Transformation: allow for custom data transformation functions

My main interest here is to learn more about hardware for machine learning. How does the interaction between the cpu and gpu really work?

Order of operations:

1. Load data
2. Apply shuffling (if any)
3. Apply transformations (if any)
4. Create batches
5. Prefetch data for next batch (while the current batch is being processed)

Update 02/07: so unfortunately since I have a Mac Intel chip, I will not be able to access the GPU and experiment with how to move data between the CPU and GPU efficiently. However, even though I am only using the CPU, I will still be able to experiment with other techniques such as prefetching. Note that the implemented prefetching will follow regular strides, i.e. consecutive memory accesses with a stride length s.
