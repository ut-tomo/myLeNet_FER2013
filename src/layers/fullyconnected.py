import numpy as np

class FullyConnected:
    def __init__(self, W, b):
        """
        Parameters:
        - W: (in_features, out_features)
        - b: (out_features,)
        """
        self.W = W
        self.b = b
        
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        x: shape (N, in_features)
        returns: shape (N, out_features)
        """
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
