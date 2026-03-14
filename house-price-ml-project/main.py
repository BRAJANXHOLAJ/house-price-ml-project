import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, r2_score

import joblib

df = pd.read_csv("data/processed/housing_clean.csv")

print(df.shape)
print(df.head())


# Baseline Model (Normal Price)


X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Save feature names for prediction interface
feature_names = X.columns.tolist()
joblib.dump(feature_names, "models/feature_names.pkl")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

lin_model = LinearRegression()

lin_model.fit(X_train, y_train)

y_pred = lin_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Performance (Normal Price)")
print("RMSE:", rmse)
print("R2:", r2)


# Method nr 2  Log transformation (reduces large errors)  This method predicts log SalePrice instead of salePrice.
# It often improves accuracy because house prices are skewed.

y_log = np.log(df["SalePrice"])

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X,
    y_log,
    test_size=0.2,
    random_state=42
)

log_model = LinearRegression()

log_model.fit(X_train_log, y_train_log)

y_pred_log = log_model.predict(X_test_log)

# convert predictions back to normal prices
y_pred_log = np.exp(y_pred_log)
y_test_actual = np.exp(y_test_log)

rmse_log = np.sqrt(mean_squared_error(y_test_actual, y_pred_log))
r2_log = r2_score(y_test_actual, y_pred_log)

print("\nLinear Regression Performance (Log Price Method)")
print("RMSE:", rmse_log)
print("R2:", r2_log)


# Decision Tree Model

tree_model = DecisionTreeRegressor(random_state=42)

tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
r2_tree = r2_score(y_test, y_pred_tree)

print("\nDecision Tree Performance")
print("RMSE:", rmse_tree)
print("R2:", r2_tree)



# Random Forest Model

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Performance")
print("RMSE:", rmse_rf)
print("R2:", r2_rf)#


# we are gona save the best model (Random Forest)

joblib.dump(rf_model, "models/random_forest_model.pkl")

print("\nBest model saved: Random Forest")

import matplotlib.pyplot as plt

# Feature importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-10:]


#result table

results = pd.DataFrame({
    "Model": ["Linear Regression", "Log Linear Regression", "Decision Tree", "Random Forest"],
    "RMSE": [rmse, rmse_log, rmse_tree, rmse_rf],
    "R2": [r2, r2_log, r2_tree, r2_rf]
})

print("\nModel Comparison")
print(results.sort_values(by="R2", ascending=False))


plt.figure(figsize=(10,6))
plt.title("Top 10 Important Features")
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Importance")
plt.show()