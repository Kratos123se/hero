import numpy as np

class LeastSquaresRegression:
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        X_transpose = np.transpose(X)
        self.coefficients = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

    def predict(self, X):
        X = np.array(X)
        return X.dot(self.coefficients)

# Example usage
X_train = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
y_train = [6, 15, 24]

# Train the model
regressor = LeastSquaresRegression()
regressor.fit(X_train, y_train)

# Predict using the trained model
X_test = [[10, 11, 12], [13, 14, 15]]
predictions = regressor.predict(X_test)

print("Predictions:", predictions)
