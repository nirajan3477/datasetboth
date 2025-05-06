import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Import SVC for classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle

# Load datasets
yield_data = pd.read_csv('synthetic_crop_yield_data.csv')
price_data = pd.read_csv('crop_prices.csv')

# Merge datasets on 'Crop'
data = pd.merge(yield_data, price_data, on='Crop')

# Prepare features and target
X = pd.get_dummies(data.drop(columns=['Yield']), columns=['Crop'])
y = data['Yield']

# Because SVM is a classifier, we need to convert the target variable 'Yield' to a categorical format.
# For demonstration, I'll use a simple approach:
# 1. Calculate the median of 'Yield'.
# 2. Create a new target variable 'y_categorical' where:
#     - If Yield >= median, y_categorical = 1 (High Yield)
#     - If Yield < median, y_categorical = 0 (Low Yield)
median_yield = y.median()
y_categorical = (y >= median_yield).astype(int)  # 1 for High, 0 for Low

# Split data
X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# 1. Support Vector Machine Classifier
svm = SVC(random_state=42)  # Initialize the SVM *Classifier*
svm.fit(X_train, y_train_cat)  # Train the model

# Save the SVM Classifier model
with open('SVMclassifier.pkl', 'wb') as f:
    pickle.dump(svm, f)

# Make predictions using SVM Classifier
preds_svm = svm.predict(X_test)

# Evaluate the SVM Classifier model. Use metrics appropriate for classification.
accuracy = accuracy_score(y_test_cat, preds_svm)
precision = precision_score(y_test_cat, preds_svm)
recall = recall_score(y_test_cat, preds_svm)
f1 = f1_score(y_test_cat, preds_svm)
try:
    auc_roc = roc_auc_score(y_test_cat, preds_svm)  # Only if you have probabilities
except ValueError:
    auc_roc = None  # roc_auc_score needs more than one class

# Print the results for SVM Classifier
print("Support Vector Machine Classifier:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
if auc_roc is not None:
    print(f"AUC-ROC: {auc_roc:.2f}")
print("-" * 20)
