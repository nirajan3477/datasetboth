import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb  # Import XGBoost
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load datasets
yield_data = pd.read_csv('synthetic_crop_yield_data.csv')
price_data = pd.read_csv('crop_prices.csv')

# Merge datasets on 'Crop'
data = pd.merge(yield_data, price_data, on='Crop')

# Prepare features and target
X = pd.get_dummies(data.drop(columns=['Yield']), columns=['Crop'])
y = data['Yield']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. XGBoost Regressor
xgbr = xgb.XGBRegressor(
    objective='reg:squarederror',  # Ensure this is set for regression
    n_estimators=100,  # You can tune this
    random_state=42
)
xgbr.fit(X_train, y_train)

# Save the XGBoost model
with open('XGboost.pkl', 'wb') as f:
    pickle.dump(xgbr, f)

# Make predictions using XGBoost
preds_xgb = xgbr.predict(X_test)

# Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, preds_xgb)
r2_xgb = r2_score(y_test, preds_xgb)

# Print the results for XGBoost Regressor
print("XGBoost Regressor:")
print(f"MSE: {mse_xgb:.2f}")
print(f"R²: {r2_xgb:.2f}")
print("-" * 20)
