# SimpleHybridRegression class (as defined previously)

class SimpleHybridRegression:
    def __init__(self, threshold=0.2, degree=3):
        self.threshold = threshold
        self.degree = degree
        self.is_polynomial = False
        self.coefficients = None

    @staticmethod
    def dot_product(a, b):
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def matrix_multiply(A, B):
        return [[sum(a * b for a, b in zip(row_a, col_b)) 
                 for col_b in zip(*B)] for row_a in A]

    @staticmethod
    def matrix_transpose(A):
        return list(map(list, zip(*A)))

    @staticmethod
    def matrix_inverse(A):
        n = len(A)
        AM = [row + [int(i == j) for j in range(n)] for i, row in enumerate(A)]
        for i in range(n):
            pivot = AM[i][i]
            AM[i] = [elem / pivot for elem in AM[i]]
            for j in range(n):
                if i != j:
                    factor = AM[j][i]
                    AM[j] = [elem - factor * AM[i][k] for k, elem in enumerate(AM[j])]
        return [row[n:] for row in AM]

    def LinearFit(self, X, y):
        X_t = self.matrix_transpose(X)
        X_t_X = self.matrix_multiply(X_t, X)
        X_t_X_inv = self.matrix_inverse(X_t_X)
        X_t_y = self.matrix_multiply(X_t, [[yi] for yi in y])
        return [coef[0] for coef in self.matrix_multiply(X_t_X_inv, X_t_y)]

    def PolynomialFit(self, X, y):
        X_poly = self.polynomial_features(X)
        return self.LinearFit(X_poly, y)

    def polynomial_features(self, X):
        return [[1] + [x[0]**d for d in range(1, self.degree+1)] for x in X]

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

    def fit(self, X, y):
        # Add bias term to X
        X = [[1] + x for x in X]

        # Fit linear regression
        linear_coef = self.LinearFit(X, y)

        # Make linear predictions
        linear_pred = [self.dot_product(x, linear_coef) for x in X]

        # Calculate MSE for linear regression
        mse_linear = self.mean_squared_error(y, linear_pred)

        if mse_linear <= self.threshold:
            self.is_polynomial = False
            self.coefficients = linear_coef
        else:
            # Fit polynomial regression
            self.is_polynomial = True
            self.coefficients = self.PolynomialFit(X, y)

    def predict(self, X):
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        X = [[1] + x for x in X]
        if self.is_polynomial:
            X = self.polynomial_features(X)

        return [self.dot_product(x, self.coefficients) for x in X]


# Define the missing evaluate_model function

from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{model_name} Results:")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    return mse, r2


# Main code to evaluate different models

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load the California Housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store results
results = {}

# SimpleHybridRegression with adjusted threshold and degree
shr = SimpleHybridRegression(threshold=0.2, degree=3)
mse_shr, r2_shr = evaluate_model(shr, X_train, X_test, y_train, y_test, "SimpleHybridRegression")
results["SimpleHybridRegression"] = {"MSE": mse_shr, "R2": r2_shr}
print(f"Used Polynomial: {shr.is_polynomial}")

# Linear Regression
lr = LinearRegression()
mse_lr, r2_lr = evaluate_model(lr, X_train, X_test, y_train, y_test, "Linear Regression")
results["Linear Regression"] = {"MSE": mse_lr, "R2": r2_lr}

# Ridge Regression
ridge = Ridge(alpha=1.0)
mse_ridge, r2_ridge = evaluate_model(ridge, X_train, X_test, y_train, y_test, "Ridge Regression")
results["Ridge Regression"] = {"MSE": mse_ridge, "R2": r2_ridge}

# Decision Tree Regression
dt = DecisionTreeRegressor(random_state=42)
mse_dt, r2_dt = evaluate_model(dt, X_train, X_test, y_train, y_test, "Decision Tree Regression")
results["Decision Tree Regression"] = {"MSE": mse_dt, "R2": r2_dt}

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)
mse_rf, r2_rf = evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest Regression")
results["Random Forest Regression"] = {"MSE": mse_rf, "R2": r2_rf}

# Compare the results
print("\nComparison:")
for model, metrics in results.items():
    print(f"{model}:")
    print(f"  MSE: {metrics['MSE']:.4f}")
    print(f"  R2 Score: {metrics['R2']:.4f}")

# Find the best model
best_model = min(results, key=lambda x: results[x]['MSE'])
print(f"\nBest model based on MSE: {best_model}")

# Feature importance for Random Forest (as an example)
feature_importance = rf.feature_importances_
feature_names = housing.feature_names
print("\nRandom Forest Feature Importances:")
for name, importance in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.4f}")

