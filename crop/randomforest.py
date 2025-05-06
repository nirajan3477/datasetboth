import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

# 2. Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)  # Initialize the model
rf.fit(X_train, y_train)  # Train the model

# Save the Random Forest model
with open('randomforest.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Make predictions using Random Forest
preds_rf = rf.predict(X_test)

# Evaluate the Random Forest model
mse_rf = mean_squared_error(y_test, preds_rf)
r2_rf = r2_score(y_test, preds_rf)

# Print the results for Random Forest
print("Random Forest Regressor:")
print(f"MSE: {mse_rf:.2f}")
print(f"R²: {r2_rf:.2f}")
print("-" * 20)
