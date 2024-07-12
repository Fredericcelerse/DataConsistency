#!/usr/bin/env python

import pandas
from sklearn.model_selection import train_test_split

# We import here our mad-home libraries
from MyPythonFunctions import (
    load_breast_cancer_data,
    display_dataset_info,
    check_data_loaded,
    preprocess_data,
    create_features,
    optimize_model,
    train_and_evaluate_model
)

# 1/ Load and pre-treat the dataset
X_breast_cancer, y_breast_cancer = load_breast_cancer_data()
display_dataset_info(X_breast_cancer, y_breast_cancer, "Breast Cancer Dataset")
check_data_loaded(X_breast_cancer, y_breast_cancer, "Breast Cancer Dataset")
X_breast_cancer_preprocessed = preprocess_data(X_breast_cancer)

# 2/ We list here the top 4 features we identified previously with combinations
top_combined_features_names_breast_cancer = [
    'worst perimeter worst smoothness',
    'worst radius worst smoothness',
    'worst texture worst concave points',
    'mean texture worst concave points',
]

# 3/ We create the new features
X_breast_cancer_enhanced = create_features(X_breast_cancer_preprocessed, combination=4)

# 4/ We extract the best features
X_breast_cancer_selected = X_breast_cancer_enhanced[top_combined_features_names_breast_cancer]

# We optimize the RF model with the 4 best features
X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_breast_cancer_selected, y_breast_cancer, test_size=0.2, random_state=42)
best_params_selected, optimization_time_selected = optimize_model(X_train_selected, y_train)

print("\nBest parameters for the RF model with only the 4 best features:")
print(best_params_selected)
print(f"Optimization time: {optimization_time_selected:.4f} seconds")

accuracy_selected, report_selected, roc_auc_selected = train_and_evaluate_model(X_train_selected, X_test_selected, y_train, y_test, best_params_selected)

print("\nOptimized model with the 4 best features:")
print(f"Accuracy: {accuracy_selected:.4f}")
print("Classification Report:")
print(report_selected)
print(f"ROC AUC Score: {roc_auc_selected:.4f}")

# 5/ Optimization of the RF model with the whole features
X_train_full, X_test_full, y_train, y_test = train_test_split(X_breast_cancer_enhanced, y_breast_cancer, test_size=0.2, random_state=42)
best_params_full, optimization_time_full = optimize_model(X_train_full, y_train)

print("\nBest parameters for the RF model with only the 4 best features:")
print(best_params_full)
print(f"Optimization time: {optimization_time_full:.4f} seconds")

accuracy_full, report_full, roc_auc_full = train_and_evaluate_model(X_train_full, X_test_full, y_train, y_test, best_params_full)

print("\nOptimized model with the whole features:")
print(f"Accuracy: {accuracy_full:.4f}")
print("Classification Report:")
print(report_full)
print(f"ROC AUC Score: {roc_auc_full:.4f}")
