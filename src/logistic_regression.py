
import numpy as np

from scripts.data.linear import linear_data, export_linear_data, read_linear_data


def train(w, b, X, y, lr=0.01, eps=1e-6):
    """
    Train the model with the given data.
    """
    
    y_pred = X @ w + b
    # apply sigmoid
    y_pred = 1 / (1 + np.exp(-y_pred) + eps)
    likelihood = y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)
    loss = -np.mean(likelihood)

    coeff_pos = X.T @ (1/(y_pred + eps) * y)
    coeff_neg = -X.T @ (1/(1-y_pred + eps) * (1-y))
    grad_w = -(coeff_pos + coeff_neg) / X.shape[0]

    coeff_pos = np.ones((y.shape[0], 1)).T @ (1/(y_pred + eps) * y)
    coeff_neg = -np.ones((y.shape[0], 1)).T @ (1/(1-y_pred + eps) * (1-y))
    grad_b = -(coeff_pos + coeff_neg) / X.shape[0]

    w -= lr * grad_w
    b -= lr * grad_b

    return w, b, loss



if __name__ == '__main__':
    
    X, y = read_linear_data('data/logistic.csv')
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    print(X.shape, y.shape)
    
    W = np.random.randn(X.shape[1], y.shape[1])
    b = np.random.randn(y.shape[1], 1)
    print(W.shape, b.shape)

    epochs = 1000
    lr = 0.01
    for i in range(epochs):
        w, b, loss = train(W, b, X, y, lr)
        print(f'Epoch {i}, loss {loss}')

        y_pred = X @ w + b
        print(y_pred.shape)



