'''
Dataloader class (non-thought out template code!!)

The main functions this dataloader class needs to support is batching, shuffling of data, and data transformations. 
'''

class Dataloader: 
    def __init__(self, X, y, batch_size, shuffle=False): 
        if shuffle: 
            self.X = self.shuffle_data(X)
        else: 
            self.X = X
        self.y = y
        self.batch_size = batch_size

    def batch_data(self): 
        assert 1 <= self.batch_size <= len(self.X)
        pass 

    def shuffle_data(self, dataset): 
        pass 

    def custom_data_transformations(self): 
        pass 
    

