# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import pickle

# Read framingham.csv into data frame
df = pd.read_csv("Data/framingham.csv")

# Feature engineering: Calculate Mean Arterial Pressure (MAP) 
df["MAP"] = df["diaBP"] + (1/3) * (df["sysBP"] - df["diaBP"]) # these features do not contain NaNs

# Undersample the majority class
chd_cases = df[df["TenYearCHD"] == 1]
non_chd_cases = df[df["TenYearCHD"] == 0].sample(n=644, random_state=42) # equal chd/non chd cases
df = pd.concat([chd_cases, non_chd_cases], axis=0)

# Drop features - determined by data_report.html
features_to_drop = ["cigsPerDay", "diabetes", "diaBP", "sysBP", "prevalentHyp", "heartRate"]
df.drop(columns=features_to_drop, inplace=True)

# Train-val-test split 
X = df.drop(columns=["TenYearCHD"])  
y = df["TenYearCHD"]
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=42)
print("Train size: ", round(X_train.shape[0] / X.shape[0], 2))
print("Validation size: ", round(X_val.shape[0] / X.shape[0], 2))
print("Test size: ", round(X_test.shape[0] / X.shape[0], 2))

# Define numerical and categorical features
num_features = ["age", "BPMeds", "totChol", "MAP", "BMI", "glucose"]
cat_features = ["male", "education", "currentSmoker", "prevalentStroke"]

# Define preprocessing steps
num_preprocessing = Pipeline([
    ("median_imputation", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_preprocessing = Pipeline([
    ("mode_imputation", SimpleImputer(strategy="most_frequent"))
])

preprocessor = ColumnTransformer([
    ("num", num_preprocessing, num_features),
    ("cat", cat_preprocessing, cat_features)
])

# Define model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(solver="liblinear", random_state=42))
])

# Hyperparameter tuning
param_grid = {
    "classifier__C": [0.01, 0.005, 0.001],  
    "classifier__penalty": ["l2"]  
} 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # stratified cross-validation 
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=3) 
grid_search.fit(X_train, y_train)

# Save X_test and y_test for later use in test.py
test_data_path = "Data/test_data.pkl"
print(f"Saving test data to {test_data_path}...")
with open(test_data_path, "wb") as pfile:
    pickle.dump((X_test, y_test), pfile)

# Save X_val and y_val for later use in test.py
val_data_path = "Data/val_data.pkl"
print(f"Saving validation data to {val_data_path}...")
with open(val_data_path, "wb") as pfile:
    pickle.dump((X_val, y_val), pfile)

# Extract best model
best_model = grid_search.best_estimator_
best_C = grid_search.best_params_["classifier__C"]
print(f"Best hyperparameter (C): {best_C}")

# Save model
pickle_path = "logistic_regression_model.pkl"
print(f"Saving model to {pickle_path}...")
with open(pickle_path, "wb") as pfile:
    pickle.dump(best_model, pfile)

# Display Feature Importance
logreg = best_model.named_steps["classifier"] # extract final model
all_feature_names = num_features + cat_features  # collect feature names
importance = np.abs(logreg.coef_).flatten() # get absolute coefficients (importance)

sorted_idx = np.argsort(importance)[::-1]
sorted_features = np.array(all_feature_names)[sorted_idx]
sorted_importance = importance[sorted_idx]

plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importance, color="blue")
plt.xlabel("Absolute Coefficient Value")
plt.ylabel("Feature")
plt.title("Feature Importance in Logistic Regression Model")
plt.gca().invert_yaxis()  
plt.savefig("Model Performance/FeatureImportance.png", bbox_inches="tight")
plt.show()

# Accuracy check
y_pred = best_model.predict(X_train)
acc = accuracy_score(y_train, y_pred, normalize=True)
print(f"Train Accuracy: {acc:.4f}")