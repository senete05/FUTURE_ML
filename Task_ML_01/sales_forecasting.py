import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# LOAD DATA
# ==========================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "senete05/ML/main/sales_forecasting_project.csv"
)

data = pd.read_csv(DATA_URL)

# ==========================================
# DATA CLEANING
# ==========================================

# Standardize region names
data["Region"] = data["Region"].str.strip().str.title()

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Handle missing values
data["Product"] = data["Product"].fillna(
    data["Product"].mode()[0]
)

data["Sales"] = data["Sales"].fillna(
    data["Sales"].mean()
)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

data["Date"] = pd.to_datetime(data["Date"])

data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day
data["DayOfWeek"] = data["Date"].dt.dayofweek

data.drop("Date", axis=1, inplace=True)

# ==========================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================

data = pd.get_dummies(
    data,
    columns=["Region", "Product"],
    drop_first=True
)

# ==========================================
# DEFINE FEATURES AND TARGET
# ==========================================

X = data.drop("Sales", axis=1)
y = data["Sales"]

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# MODEL TRAINING
# ==========================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 30)
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

importance = importance.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("-" * 30)
print(importance)