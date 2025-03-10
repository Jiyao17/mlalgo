


import numpy as np


def linear_data(n: int, shape: tuple[int], noise=0.1) -> np.ndarray:
    """
    Generate linear data with n samples and shape features.
    """
    X = np.random.randn(n, shape)
    w = np.random.randn(shape)
    b = np.random.randn()
    y = np.dot(X, w) + b + np.random.randn(n) * noise
    return X, y, w, b

def export_linear_data(X, y, path):
    # export to csv, each row is a sample
    data = np.concatenate([X, y[:, None]], axis=1)
    data = data.round(5)
    np.savetxt(path, data, delimiter=',', fmt='%.5f')


def read_linear_data(path):
    """
    Read linear data from a file.
    """
    data = np.loadtxt(path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


if __name__ == '__main__':
    
    X, y, w, b = linear_data(1000, 10)
    export_linear_data(X, y, 'data/linear.csv')

    X, y = read_linear_data('data/linear.csv')
    print(X.shape, y.shape)

    # make y = 1 for positive values, 0 otherwise
    y = (y > 0).astype(int)
    export_linear_data(X, y, 'data/logistic.csv')
    X, y = read_linear_data('data/logistic.csv')
    print(X.shape, y.shape)
    
