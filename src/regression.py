
import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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


def my_regression(X_train, y_train, X_test, y_test):
    epochs = 100
    lr = 0.1
    for i in range(epochs):
        w, b, loss = train(W, b, X_train, y_train, lr)
        # print(f'Epoch {i}, loss {loss}')

    return w, b


def sklearn_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(mean_squared_error(y_test, y_pred))


if __name__ == '__main__':
    X, y = read_linear_data('data/linear.csv')
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    print(X.shape, y.shape)
    
    w0 = np.random.randn(X.shape[1], y.shape[1])
    b0 = np.random.randn(y.shape[1], 1)
    print(w0.shape, b0.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    w, b = w0.copy(), b0.copy()
    for i in range(100):
        # vary lr to see the difference from sklearn
        w, b, loss = train(w, b, X_train, y_train, 0.1)
    
    y_pred = X_test @ w + b
    print(mean_squared_error(y_test, y_pred))

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print(mean_squared_error(y_test, y_pred))