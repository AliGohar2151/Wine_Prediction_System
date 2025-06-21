import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

print("Starting model training script...")


try:
    wine_df = pd.read_csv("winequality-red (1).csv", sep=";")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(
        "Error: 'winequality-red (1).csv' not found. Please ensure it's in the same directory."
    )
    exit()


X = wine_df.drop("quality", axis=1)
y = wine_df["quality"]

print(f"Original quality values unique: {np.sort(y.unique())}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("Data split into training and testing sets.")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled using StandardScaler.")


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("RandomForestRegressor model trained successfully.")

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Evaluation (on test set):")
print(f"  Mean Squared Error (MSE): {mse:.4f}")
print(f"  R-squared (R2): {r2:.4f}")

model_filename = "wine_model_regression.pkl"
scaler_filename = "scaler.pkl"

with open(model_filename, "wb") as f:
    pickle.dump(model, f)
print(f"Regression model saved as '{model_filename}'")

with open(scaler_filename, "wb") as f:
    pickle.dump(scaler, f)
print(f"Scaler saved as '{scaler_filename}'")

print("Model training script finished.")
