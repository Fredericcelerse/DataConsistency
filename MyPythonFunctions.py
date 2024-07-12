#!/usr/bin/env python

# We import the libraries that the functions need 
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Function to load the dataset "breasr_cancer" from the scikit-learn library
def load_breast_cancer_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y

# Function to visualize the dataset
def display_dataset_info(X, y, dataset_name):
    print(f"Dataset: {dataset_name}")
    print(f"Features: {list(X.columns)}")
    print(f"Shape: {X.shape}")
    print("First 5 rows of the dataset:")
    print(X.head())
    print("\n")

# Function to check if the dataset was correctly loaded
def check_data_loaded(X, y, dataset_name):
    if X is not None and y is not None and not X.empty and not y.empty:
        print(f"{dataset_name} loaded successfully.")
    else:
        print(f"Error loading {dataset_name}.")

# Function to preprocess the data
def preprocess_data(X):
    X = X.copy()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].fillna('missing')
        X[col] = LabelEncoder().fit_transform(X[col])
    for col in X.select_dtypes(include=['float64', 'int64']).columns:
        X[col] = X[col].fillna(X[col].median())
    return X

# Function to create interactions between the features and apply the non-linear transformations
def create_features(X, combination):
    poly = PolynomialFeatures(degree=combination, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X)
    return pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

# Function to compute the correlations between the new features and the corresponded target
def calculate_correlations(X, y):
    correlations = {}
    for col in X.columns:
        corr, _ = pearsonr(X[col], y)
        correlations[col] = abs(corr)
    return correlations

# Function to visualize the best features
def visualize_top_features(X, y, top_features, dataset_name):
    top_features_df = X[top_features]
    top_features_df['target'] = y
    sns.pairplot(top_features_df, hue='target')
    plt.suptitle(f'Relationships in {dataset_name}', y=1.02)
    plt.show()

# Function to train a Random Forest classification algorithm
def train_RF(X_train, X_test, y_train, y_test):
    start_time = time.time()
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return training_time, accuracy

# Function to automatized the hyperparameters tuning for RF
def optimize_model(X_train, y_train):
    param_space = {
        'n_estimators': (100, 300),
        'max_depth': (10, 30),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4)
    }
    
    rf = RandomForestClassifier(random_state=42)
    bayes_search = BayesSearchCV(estimator=rf, search_spaces=param_space, cv=5, n_jobs=-1, verbose=2, n_iter=50)
    
    start_time = time.time()
    bayes_search.fit(X_train, y_train)
    optimization_time = time.time() - start_time
    
    return bayes_search.best_params_, optimization_time

# Function to retrain and evaluate RF model with the best parameters
def train_and_evaluate_model(X_train, X_test, y_train, y_test, best_params):
    model = RandomForestClassifier(random_state=42, **best_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return accuracy, report, roc_auc
