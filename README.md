f# ML From Scratch

This is my implementation of (feed forward) neural networks from scratch. The point is to learn how some of these technologies work under the hood! The reach goal is to code my own transformer. This is going to be challenging ;)) Note: I also want this to be a project that follows decent enough coding practices, not necessarily with unit tests but maybe some asserts throughout. ALSO! None of the code in this repository was copied from existing repositories or was generated by any form of generative AI (including copilot) ~~ this will be an old school programming project to prove to myself that I can still function and think without it!

## Essential modules

- Dataloader: a module that is able to process batches of data at a time. I want this to be optimized with prefetching so that we limit the out of memory issues.
- Layers: a module with different types of neural network layers to construct customizable models (and architectures).
  - Dense: Implements dense layers. It will need to support both regression and classification tasks.
- Compile: a module that constructs the computational graph for the network.
- More TBD

**Contributors:** _Christian Aagnes_
