import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)

    m, n = X.shape

    # Initialize parameters
    w = np.zeros(n)
    b = 0

    for _ in range(steps):
        # Linear combination
        z = np.dot(X, w) + b
        
        # Prediction using sigmoid
        y_pred = _sigmoid(z)
        
        # Gradients
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db

    return w, b
    pass