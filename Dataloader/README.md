# Dataloader

This module implements the dataloader objects, that will serve as the inputs to our neural networks.

Features that I want it to support:

- Batching: The most important is that it is an iterable interface, such that it is easy to loop through the dataset in batches (of custom size).
- Shuffling: randomizes the order of data samples (to not overfit to sequential patterns in time series) - should be a flag that can be set to True/False
- Data Transformation: allow for custom data transformation functions?

My main interest here is to learn more about hardware for machine learning. How does the interaction between the cpu and gpu really work?
