import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

class SimpleHybridRegression:
    def __init__(self, threshold=0.5, degree=2):
        self.threshold = threshold
        self.degree = degree
        self.is_polynomial = False
        self.model = None
        self.coefficients = None

    def fit(self, X, y):
        # Convert to numpy arrays for easier manipulation
        X = np.array(X)
        y = np.array(y)

        # Fit linear regression
        linear_model = LinearRegression()
        linear_model.fit(X, y)

        # Make linear predictions
        linear_pred = linear_model.predict(X)

        # Calculate MSE for linear regression
        mse_linear = mean_squared_error(y, linear_pred)

        if mse_linear <= self.threshold:
            self.is_polynomial = False
            self.model = linear_model
            self.coefficients = linear_model.coef_
        else:
            # Fit polynomial regression
            self.is_polynomial = True
            poly_features = PolynomialFeatures(degree=self.degree)
            X_poly = poly_features.fit_transform(X)
            polynomial_model = LinearRegression()
            polynomial_model.fit(X_poly, y)
            self.model = polynomial_model
            self.coefficients = polynomial_model.coef_

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        X = np.array(X)
        if self.is_polynomial:
            poly_features = PolynomialFeatures(degree=self.degree)
            X = poly_features.fit_transform(X)

        return self.model.predict(X)

# Example usage
if __name__ == "__main__":
    X = [[1], [2], [3], [4], [5]]
    y = [2, 4, 5, 4, 5]
    threshold = 0.5
    degree = 2

    model = SimpleHybridRegression(threshold, degree)
    model.fit(X, y)
    predictions = model.predict(X)

    print(f"Predictions: {predictions}")
    print(f"Used polynomial regression: {model.is_polynomial}")
    print(f"Coefficients: {model.coefficients}")

    # Calculate MSE
    mse = mean_squared_error(y, predictions)
    print(f"Mean Squared Error: {mse}")

