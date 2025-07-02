'''
Dataloader class (non-thought out template code!!)

X: independent variable stored as a numpy array, shape: (batch size, feature size)
y: response variable stored as a numpy array, shape: (feature size, )
'''

import numpy as np


class Dataloader: 
    def __init__(self, X, y, batch_size, shuffle=False, transformations=None): 
        self.X = X 
        self.y = y.reshape(1, -1)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transformations = transformations

        if self.shuffle: 
            self.X, self.y = self.shuffle_data()
        if transformations: 
            self.X = self.custom_data_transformations()


    def shuffle_data(self): 
        assert X.shape[1] == y.shape[1], "X and y do not have matching dimensions. X should be a 2D numpy array of size (batch size, feature size) while y should be a 1D numpy array of size (feature size, )."
        shuffler = np.random.default_rng()
        full_data = np.concatenate(self.X, self.y) # make sure (X, y) pairs get shuffled together
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
        assert 1 <= self.batch_size <= len(self.X)
        pass 


if __name__=="__main__":
    X = np.array([[1, 2, 3, 4, 5], [-10, -20, -30, -40, -50]])
    y = np.array([15, 25, 35, 45, 55]).reshape(1, -1)
    def myfunc(x):
        return x+2
    #print(np.vectorize(myfunc)(X))
    print(X.shape, y.shape)
    full = np.concatenate((X, y))
    print(full)
    print("")

    shuffler = np.random.default_rng()
    shuffler.shuffle(full, axis=1)
    print(full)
    
    print("")
    X, y = full[:-1, :], full[-1, :]
    print(X, y)

    assert callable(myfunc)
    assert X.shape[1] == y.shape[1] 