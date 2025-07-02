# Dataloader

This module implements the dataloader objects, that will serve as the inputs to our neural networks.

Features that I want it to support:

- Batching: The most important is that it is an iterable interface, such that it is easy to loop through the dataset in batches (of custom size).
- Shuffling: randomizes the order of data samples (to not overfit to sequential patterns in time series)
- Data Transformation: allow for custom data transformation functions

My main interest here is to learn more about hardware for machine learning. How does the interaction between the cpu and gpu really work?

Update: so unfortunately since I have a Mac Intel chip, I will not be able to access the GPU and experiment with how to move data between the CPU and GPU efficiently. However, even though I am only using the CPU, I will still be able to experiment with other techniques such as prefetching.
