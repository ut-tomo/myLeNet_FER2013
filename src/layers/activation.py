import numpy as np

class Tanh:
    def forward(self, x):
        return np.tanh(x)

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
class Softmax:
    def forward(self, x):
        if x.ndim == 2:
            x_max = np.max(x, axis=1, keepdims=True)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        elif x.ndim == 1:
            x_max = np.max(x)
            exp_x = np.exp(x - x_max)
            return exp_x / np.sum(exp_x)
        else:
            raise ValueError("Invalid input shape: expected 1D or 2D array, got shape {}".format(x.shape))