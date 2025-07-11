'''
Dataloader class (non-thought out template code!!)

X: independent variable stored as a numpy array, shape: (batch size, feature size)
y: response variable stored as a numpy array, shape: (feature size, )
batch_size: number of data points used for gradient updates, defaults to 1. Note: the last batch will be dropped.
'''

import prefetch
import numpy as np
from math import floor


class Dataloader: 
    def __init__(self, X, y, batch_size=1, prefetch=True, shuffle=False, transformations=None): 
        self.X = X 
        self.y = y.reshape(1, -1)
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.shuffle = shuffle
        self.transformations = transformations

        if self.shuffle: 
            self.X, self.y = self.shuffle_data()
        if transformations: 
            self.X = self.custom_data_transformations()

        self.batches = self.batch_data()
        

    def shuffle_data(self): 
        assert self.X.shape[1] == self.y.shape[1], "X and y do not have matching dimensions. X should be a 2D numpy array of size (batch size, feature size) while y should be a 1D numpy array of size (feature size, )."
        shuffler = np.random.default_rng()
        full_data = np.concatenate((self.X, self.y)) # make sure (X, y) pairs get shuffled together
        shuffler.shuffle(full_data, axis=1)  # shuffles the columns the combined data
        self.X, self.y = full_data[:-1, :], full_data[-1, :]
        return self.X, self.y


    def custom_data_transformations(self): # list 
        assert isinstance(self.transformations, list), "Transformations must be a list of functions."
        for t in self.transformations: 
            assert callable(t), f"{t} is not a callable function."
            self.X = np.vectorize(t)(self.X) 
        return self.X


    def batch_data(self): 
        assert 1 <= self.batch_size <= self.X.shape[0], "The batch size needs to be a positive integer between 1 and the length of the dataset (inclusive)."
        
        batched_data = []
        for start in range(0, self.X.shape[0], self.batch_size): # drops remaining data samples
            end = start + self.batch_size 
            if end > self.X.shape[0]:
                break 
            
            batched_data.append((self.X[start:end], self.y[0][start:end]))

            """
            if self.prefetch: # call prefetch.cpp
                prefetch.process_batch(X_batch, y_batch) 
                yield X_batch, y_batch
            else: # store batches without prefetching
                pass
            """

        return batched_data

if __name__=="__main__":
    X = np.array([[1, 2, 3, 4, 5], [-10, -20, -30, -40, -50], 
                  [1, 2, 3, 4, 5], [-10, -20, -30, -40, -50]])
    y = np.array([15, 25, 35, 45, 55])
    def add2(x):
        return x+2
    def del3(x):
        return x-3
    
    test = Dataloader(X, y, batch_size=3, prefetch=True, shuffle=False, transformations=[])
    
    #print("")
    #print(X)
    #print(test.X, "\n")

    print(y)
    print(y[0:2])
    print(test.y, "\n")

    print("")
    print(test.batches)

    print("")