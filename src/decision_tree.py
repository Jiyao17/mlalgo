


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from scripts.data.linear import linear_data, export_linear_data, read_linear_data



if __name__ == '__main__':
    
    X, y = read_linear_data('data/logistic.csv')
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    print(X.shape, y.shape)
    y = y.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    
