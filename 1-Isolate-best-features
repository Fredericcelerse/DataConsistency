#!/usr/bin/env python

import pandas as pd

# We import here our mad-home libraries
from MyPythonFunctions import (
    load_breast_cancer_data,
    display_dataset_info,
    check_data_loaded,
    preprocess_data,
    create_features,
    calculate_correlations,
    visualize_top_features
)

# 1/ First, load the dataset and pre-treat the data
X_breast_cancer, y_breast_cancer = load_breast_cancer_data()
display_dataset_info(X_breast_cancer, y_breast_cancer, "Breast Cancer Dataset")
X_breast_cancer_preprocessed = preprocess_data(X_breast_cancer)

# 2/ Check the loading of the data
check_data_loaded(X_breast_cancer, y_breast_cancer, "Breast Cancer Dataset")

# 3/ Then, create the new features
X_breast_cancer_enhanced = create_features(X_breast_cancer_preprocessed, combination=4)	# combination here define the degree of combination we will use

# 4/ Compute the correlations between the new features and the target
correlations_breast_cancer_enhanced = calculate_correlations(X_breast_cancer_enhanced, y_breast_cancer)
top_features_breast_cancer = sorted(correlations_breast_cancer_enhanced.items(), key=lambda item: item[1], reverse=True)[:10]

# 5/ Print the 10 best features with their score
print("Top 10 features with their correlation score -- Breast Cancer Dataset:")
for feature, score in top_features_breast_cancer:
    print(f"{feature}: {score}")

# 6/ Extract the name of the best features
top_features_names_breast_cancer = [feature for feature, _ in top_features_breast_cancer]

# 7/ Optionnaly, visualize their relationship with the target
#visualize_top_features(X_breast_cancer_enhanced, y_breast_cancer, top_features_names_breast_cancer, "Breast Cancer Dataset Enhanced")

