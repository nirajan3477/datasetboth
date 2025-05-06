import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  # Import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
from sklearn.preprocessing import LabelEncoder

# Load datasets
yield_data = pd.read_csv('synthetic_crop_yield_data.csv')
price_data = pd.read_csv('crop_prices.csv')

# Merge datasets on 'Crop'
data = pd.merge(yield_data, price_data, on='Crop')

# Prepare features and target
X = pd.get_dummies(data.drop(columns=['Yield']), columns=['Crop'])
y = data['Yield']

# Because Naive Bayes is a classifier, we need to convert the target variable 'Yield' to a categorical format.
#  For demonstration, I'll use a simple approach:
#  1. Calculate the median of 'Yield'.
#  2.  Create a new target variable 'y_categorical' where:
#      - If Yield >= median, y_categorical = 1 (High Yield)
#      - If Yield < median, y_categorical = 0 (Low Yield)
median_yield = y.median()
y_categorical = (y >= median_yield).astype(int)  # 1 for High, 0 for Low

# Split data
X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# 1. Gaussian Naive Bayes
nb = GaussianNB()  # Initialize the model
nb.fit(X_train, y_train_cat)  # Train the model

# Save the Naive Bayes model
with open('NBclassifier.pkl', 'wb') as f:
    pickle.dump(nb, f)

# Make predictions using Naive Bayes
preds_nb = nb.predict(X_test)

# Evaluate the Naive Bayes model.  Use metrics appropriate for classification.
accuracy = accuracy_score(y_test_cat, preds_nb)
precision = precision_score(y_test_cat, preds_nb)
recall = recall_score(y_test_cat, preds_nb)
f1 = f1_score(y_test_cat, preds_nb)
try:
    auc_roc = roc_auc_score(y_test_cat, preds_nb)  # Only if you have probabilities
except ValueError:
    auc_roc = None # roc_auc_score needs more than one class

# Print the results for Naive Bayes
print("Gaussian Naive Bayes Classifier:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
if auc_roc is not None:
    print(f"AUC-ROC: {auc_roc:.2f}")
print("-" * 20)
