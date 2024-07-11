#!/usr/bin/env python

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# We import here our mad-home libraries
from MyPythonFunctions import (
    load_breast_cancer_data,
    display_dataset_info,
    check_data_loaded,
    preprocess_data,
    create_features,
    train_RF
)

# 1/ Load and pre-treat the dataset
X_breast_cancer, y_breast_cancer = load_breast_cancer_data()
display_dataset_info(X_breast_cancer, y_breast_cancer, "Breast Cancer Dataset")
check_data_loaded(X_breast_cancer, y_breast_cancer, "Breast Cancer Dataset")
X_breast_cancer_preprocessed = preprocess_data(X_breast_cancer)

# 2/ We list here the top 10 features we identified previously without combination
top_features_names_breast_cancer = [
    'worst concave points',
    'worst perimeter',
    'mean concave points',
    'worst radius',
    'mean perimeter',
    'worst area',
    'mean radius',
    'mean area',
    'mean concavity',
    'worst concavity'
]

# 3/ And now the top 10 features we identified previously with combinations
top_combined_features_names_breast_cancer = [
    'worst perimeter worst smoothness',
    'worst radius worst smoothness',
    'worst texture worst concave points',
    'mean texture worst concave points',
    'worst radius worst concave points',
    'mean radius worst texture worst concave points',
    'mean radius worst concave points',
    'mean concave points worst texture',
    'mean perimeter worst texture worst concave points',
    'mean perimeter worst smoothness'
]

# 4/ We create the same new features as before
X_breast_cancer_enhanced1 = create_features(X_breast_cancer_preprocessed, combination=1)
X_breast_cancer_enhanced2 = create_features(X_breast_cancer_preprocessed, combination=4)

# 5/ We extract the best features
X_breast_cancer_selected1 = X_breast_cancer_enhanced1[top_features_names_breast_cancer]
X_breast_cancer_selected2 = X_breast_cancer_enhanced2[top_combined_features_names_breast_cancer]

# 6/ Now, we build our ML model

# => We train a classical Random Forest model only with the best non-combined features
feature_counts = list(range(1, X_breast_cancer_selected1.shape[1] + 1, 1))
training_times1 = []
accuracies1 = []
for i, count in enumerate(feature_counts):

    X_subset = X_breast_cancer_selected1.iloc[:, :count]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_breast_cancer, test_size=0.2, random_state=42)

    t, acc = train_RF(X_train, X_test, y_train, y_test)

    training_times1.append(t)
    accuracies1.append(acc)

    print("{}/{} done !".format(i+1,len(feature_counts)))
    time.sleep(1)

# => we print the training time and accuracy for each of these trainings
print("\nCASE1: THE BEST NON_COMBINED FEATURES\n")
for i, j in enumerate(training_times1):
    print("Number of features: {} --- Training time: {} --- Accuracy: {}".format(i+1, j, accuracies1[i]))

# => Finally, we repeat the process for the other cases
# CASE 2: All the non-combined data
X_train, X_test, y_train, y_test = train_test_split(X_breast_cancer_preprocessed, y_breast_cancer, test_size=0.2, random_state=42)

t, acc = train_RF(X_train, X_test, y_train, y_test)

print("\nCASE2: THE BEST NON_COMBINED FEATURES\n")
print("Number of features: {} --- Training time: {} --- Accuracy: {}".format(len(X_breast_cancer_enhanced1), t, acc))

# CASE 3: The 10 best combined features
training_times1 = []
accuracies1 = []
for i, count in enumerate(feature_counts):

    X_subset = X_breast_cancer_selected2.iloc[:, :count]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_breast_cancer, test_size=0.2, random_state=42)

    t, acc = train_RF(X_train, X_test, y_train, y_test)

    training_times1.append(t)
    accuracies1.append(acc)

    print("{}/{} done !".format(i+1,len(feature_counts)))
    time.sleep(1)

# => we print the training time and accuracy for each of these trainings
print("\nCASE3: THE BEST NON_COMBINED FEATURES\n")
for i, j in enumerate(training_times1):
    print("Number of features: {} --- Training time: {} --- Accuracy: {}".format(i+1, j, accuracies1[i]))

# CASE 4: All the combined features
X_train, X_test, y_train, y_test = train_test_split(X_breast_cancer_enhanced2, y_breast_cancer, test_size=0.2, random_state=42)

t, acc = train_RF(X_train, X_test, y_train, y_test)

print("\nCASE4: THE BEST NON_COMBINED FEATURES\n")
print("Number of features: {} --- Training time: {} --- Accuracy: {}".format(len(X_breast_cancer_enhanced2), t, acc))
