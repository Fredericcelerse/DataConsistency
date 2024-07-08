#!/usr/bin/env python

# We first read the data
import pandas as pd
drug_data_path = "databases/drug200.csv"
cancer_data_path = "databases/lung_cancer.csv"

drug_data = pd.read_csv(drug_data_path)
cancer_data = pd.read_csv(cancer_data_path)

# We check the data and separate the features to the target

print(drug_data.head())
print(cancer_data.head())

X_drug = drug_data.iloc[:, :-1]
y_drug = drug_data.iloc[:, -1]

X_cancer = cancer_data.iloc[:, :-1]
y_cancer = cancer_data.iloc[:, -1]

# We convert the target into binaries
from sklearn.preprocessing import LabelEncoder

label_encoder_drug = LabelEncoder()
y_drug = label_encoder_drug.fit_transform(y_drug)

label_encoder_cancer = LabelEncoder()
y_cancer = label_encoder_cancer.fit_transform(y_cancer)

# We pre-treat the features before using them

def preprocess_data(X):
    X = X.copy()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].fillna('missing')
        X[col] = LabelEncoder().fit_transform(X[col])
    for col in X.select_dtypes(include=['float64', 'int64']).columns:
        X[col] = X[col].fillna(X[col].median())
    return X

X_drug_preprocessed = preprocess_data(X_drug)
X_cancer_preprocessed = preprocess_data(X_cancer)

# In order to visualize how the features can correlate well with the target, 
# we create interactions between the features and then we apply non-linear
# tranformations
from sklearn.preprocessing import PolynomialFeatures

def create_features(X):
    poly = PolynomialFeatures(degree=4, interaction_only=False, include_bias=False)  # 4 here, but we can choose lower or higher type of combinations
    X_poly = poly.fit_transform(X)
    return pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

X_drug_enhanced = create_features(X_drug_preprocessed)
X_cancer_enhanced = create_features(X_cancer_preprocessed)

# Once the new correlated features are created, we can calculate the correlation
# using the Pearson Correlation method 
from scipy.stats import pearsonr

def calculate_correlations(X, y):
    correlations = {}
    for col in X.columns:
        corr, _ = pearsonr(X[col], y)
        correlations[col] = abs(corr)
    return correlations

correlations_drug_enhanced = calculate_correlations(X_drug_enhanced, y_drug)
print("Correlations with the target -> drug200.csv:")
print(sorted(correlations_drug_enhanced.items(), key=lambda item: item[1], reverse=True)[:5])  # Print only the 5 best correlations

correlations_cancer_enhanced = calculate_correlations(X_cancer_enhanced, y_cancer)
print("Correlations with the target -> dataset.csv:")
print(sorted(correlations_cancer_enhanced.items(), key=lambda item: item[1], reverse=True)[:5]) # Print only the 5 best correlations
