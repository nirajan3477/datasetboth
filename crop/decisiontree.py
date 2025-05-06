import pandas as pd
from sklearn.tree import DecisionTreeRegressor
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

# 1. Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# Save the model
with open('decisiontree.pkl', 'wb') as f:
    pickle.dump(dt, f)

# Make predictions
preds_dt = dt.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, preds_dt)
r2 = r2_score(y_test, preds_dt)

# Print the results
print("Decision Tree Regressor:")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")
print("-" * 20)
