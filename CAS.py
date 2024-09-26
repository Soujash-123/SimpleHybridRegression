from SimpleHybridRegression import SimpleHybridRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{model_name} Results:")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    return mse, r2

# Load the California Housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results
results = {}

# SimpleHybridRegression
shr = SimpleHybridRegression(threshold=0.5, degree=2)
mse_shr, r2_shr = evaluate_model(shr, X_train_scaled, X_test_scaled, y_train, y_test, "SimpleHybridRegression")
results["SimpleHybridRegression"] = {"MSE": mse_shr, "R2": r2_shr}
print(f"Used Polynomial: {shr.is_polynomial}")

# Linear Regression
lr = LinearRegression()
mse_lr, r2_lr = evaluate_model(lr, X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression")
results["Linear Regression"] = {"MSE": mse_lr, "R2": r2_lr}

# Ridge Regression
ridge = Ridge(alpha=1.0)
mse_ridge, r2_ridge = evaluate_model(ridge, X_train_scaled, X_test_scaled, y_train, y_test, "Ridge Regression")
results["Ridge Regression"] = {"MSE": mse_ridge, "R2": r2_ridge}

# Decision Tree Regression
dt = DecisionTreeRegressor(random_state=42)
mse_dt, r2_dt = evaluate_model(dt, X_train_scaled, X_test_scaled, y_train, y_test, "Decision Tree Regression")
results["Decision Tree Regression"] = {"MSE": mse_dt, "R2": r2_dt}

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=42)
mse_rf, r2_rf = evaluate_model(rf, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest Regression")
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
