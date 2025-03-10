
import numpy as np

from scripts.data.linear import linear_data, export_linear_data, read_linear_data


def train(w, b, X, y, lr=0.01):
    """
    Train the model with the given data.
    """
    
    y_pred = X @ w + b
    loss = np.mean((y - y_pred) ** 2)

    grad_w = 2 * -X.T @ (y - y_pred) / X.shape[0]
    grad_b = 2 * -np.ones((X.shape[0], 1)).T @ (y - y_pred) / X.shape[0]

    w -= lr * grad_w 
    b -= lr * grad_b

    return w, b, loss



if __name__ == '__main__':
    
    X, y = read_linear_data('data/linear.csv')
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    print(X.shape, y.shape)
    
    W = np.random.randn(X.shape[1], y.shape[1])
    b = np.random.randn(y.shape[1], 1)
    print(W.shape, b.shape)

    epochs = 100
    lr = 0.1
    for i in range(epochs):
        w, b, loss = train(W, b, X, y, lr)
        print(f'Epoch {i}, loss {loss}')

        y_pred = X @ w + b
        print(y_pred.shape)



