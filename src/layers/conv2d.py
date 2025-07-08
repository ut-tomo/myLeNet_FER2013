import numpy as np
import sys, os 
sys.path.append(os.pardir)
from utils.utils import im2col


class Conv2D:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
    def forward(self, x):
        N, C_in, H_in, W_in = x.shape
        FN, _, FH, FW = self.W.shape
        
        out_h = int(1 + (H_in + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W_in + 2 * self.pad - FW) / self.stride)
        
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        
        out = out.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        
        return out
    
