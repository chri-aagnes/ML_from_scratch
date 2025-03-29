This is my implementation of (feed forward) neural networks from scratch. The point is to learn how some of these technologies work under the hood! The reach goal is to code my own transformer. This is going to be challenging ;))

Essential parts/modules:

- Dataloader: a module that is able to process batches of data at a time. I want this to be optimized with prefetching so that we limit the out of memory issues.
- FFNN: this will be the core module for training a feed forward neural network. It will need to support both regression and classification tasks.
- More TBD
