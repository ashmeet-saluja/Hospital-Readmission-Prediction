#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
file_path = 'diabetic_data.csv'
data = pd.read_csv(file_path)

# Step 2: Data Cleaning
# Replace '?' with NaN and drop unnecessary columns
data_cleaned = data.replace('?', pd.NA)
columns_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty']
data_cleaned = data_cleaned.drop(columns=columns_to_drop)

# Drop rows with missing values
data_cleaned = data_cleaned.dropna()

# Encode categorical variables
categorical_cols = data_cleaned.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data_cleaned[col] = le.fit_transform(data_cleaned[col])
    label_encoders[col] = le

# Step 3: Sample the data for faster processing (20% of the data)
data_sampled = data_cleaned.sample(frac=0.2, random_state=42)

# Step 4: Split the data into features and target variable
X_sampled = data_sampled.drop('readmitted', axis=1)
y_sampled = data_sampled['readmitted']

# Train-Test Split
X_train_sampled, X_test_sampled, y_train_sampled, y_test_sampled = train_test_split(
    X_sampled, y_sampled, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_sampled, y_train_sampled)

# Step 6: Make Predictions and Evaluate the Model
y_pred_sampled = rf_model.predict(X_test_sampled)
accuracy_sampled = accuracy_score(y_test_sampled, y_pred_sampled)
report_sampled = classification_report(y_test_sampled, y_pred_sampled)

# Print the results
print("Accuracy:", accuracy_sampled)
print("Classification Report:\n", report_sampled)


# In[3]:


from sklearn.model_selection import GridSearchCV

# Step 7: Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Step 8: Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    verbose=2,
    n_jobs=-1
)

# Fit the grid search to the training data
grid_search.fit(X_train_sampled, y_train_sampled)

# Get the best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Step 9: Evaluate the tuned model on the test set
y_pred_best = best_model.predict(X_test_sampled)
accuracy_best = accuracy_score(y_test_sampled, y_pred_best)
report_best = classification_report(y_test_sampled, y_pred_best)

# Print results
print("Best Parameters:", best_params)
print("Accuracy After Tuning:", accuracy_best)
print("Classification Report After Tuning:\n", report_best)


# In[4]:


import matplotlib.pyplot as plt
import numpy as np

# Get feature importances from the best model
feature_importances = best_model.feature_importances_
sorted_idx = np.argsort(feature_importances)

# Plot the feature importances
plt.figure(figsize=(10, 8))
plt.barh(X_train_sampled.columns[sorted_idx], feature_importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Random Forest")
plt.show()


# In[5]:


import joblib

# Save the trained model
joblib.dump(best_model, 'hospital_readmission_model.pkl')

# Save the label encoders for categorical variables
joblib.dump(label_encoders, 'label_encoders.pkl')


# In[6]:


# Load the model and encoders
loaded_model = joblib.load('hospital_readmission_model.pkl')
loaded_encoders = joblib.load('label_encoders.pkl')

# Test the loaded model on a new data point
sample_data = X_test_sampled.iloc[0].values.reshape(1, -1)  # Take one test sample
prediction = loaded_model.predict(sample_data)

# Decode the prediction
decoded_prediction = list(loaded_encoders['readmitted'].inverse_transform(prediction))[0]
print("Predicted Readmission Category:", decoded_prediction)


# In[ ]:




