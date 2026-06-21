import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # Forward Pass
        z1 = np.dot(x, W1.T) + b1
        a1 = np.maximum(0, z1)
        y_hat = np.dot(a1, W2.T) + b2

        # Loss - Mean Squared Error
        error = y_hat - y_true
        L = np.round(np.mean(np.square(error)),4)
        n = y_true.size

        # Gradients
        out_gradient = 2 * error / n
        relu_gradient = W2 * (z1 > 0)
      
        dW1 = np.round((out_gradient * relu_gradient).T * x, 4)
        db1 = np.round(out_gradient * relu_gradient.flatten(), 4)
        dW2 = np.round(out_gradient.reshape(-1,1) * a1, 4)
        db2 = np.round(out_gradient, 4)

        return {
            'loss' : L,
            'dW1' : dW1.tolist(),
            'db1' : db1.tolist(),
            'dW2' : dW2.tolist(),
            'db2' : db2.tolist()
        }

