import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        
        n_feat = X.shape[1]
        w = np.zeros(n_feat)
        b = 0

        for _ in range(epochs):
            # Linear Regression
            y_hat = np.dot(X, w) + b

            # Loss - Mean Squared Error
            error = y_hat - y
            L = np.mean(np.square(error))
            
            # Gradients
            n = y.size
            dw = (2 / n) * np.dot(X.T, error) 
            db = 2 * np.mean(error)

            # Weight Update
            w -= lr * dw
            b -= lr * db 

        return (np.round(w, 5), round(b, 5))