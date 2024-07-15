#!/usr/bin/env python

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import pearsonr

anemia_data_path = "d_output.csv"
anemia_data = pd.read_csv(anemia_data_path)

# We check the data and separate the features from the target
print(anemia_data.head())

# We store in two variables the features and the target
X_features = anemia_data.iloc[:, :-1]
y_anemia = anemia_data.iloc[:, -1]

# First, we convert the target into binaries
label_encoder_anemia = LabelEncoder()
y_anemia = label_encoder_anemia.fit_transform(y_anemia)

# Then, we process the initial features
def preprocess_data(X):
    X = X.copy()
    label_encoder = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].fillna('missing')
        X[col] = label_encoder.fit_transform(X[col])
    for col in X.select_dtypes(include=['float64', 'int64']).columns:
        X[col] = X[col].fillna(X[col].median())
    return X

X_features_preprocessed = preprocess_data(X_features)

# Standardize the features
scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X_features_preprocessed)

# We create different combinations of new features by creating interactions between the features and then we apply non-linear transformations
def create_features(X, combination, feature_names):
    poly = PolynomialFeatures(degree=combination, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X)
    return pd.DataFrame(X_poly, columns=poly.get_feature_names_out(feature_names))

# => As we have only 5 features here, we can create all the possible combination of features easily
store_new_features = []
for i in range(5):
    X_features_enhanced = create_features(X_features_scaled, i+1, X_features_preprocessed.columns)
    store_new_features.append(X_features_enhanced)
    print("New combined features with combination = {} created !".format(i+1))

# For each of the new features, we compute their correlation with the respective target

def calculate_correlations(X, y):
    correlations = {}
    for col in X.columns:
        corr, _ = pearsonr(X[col], y)
        correlations[col] = abs(corr)
    return correlations

best_correlations = None
best_features = None

for i in range(5):
    correlations = calculate_correlations(store_new_features[i], y_anemia)
    print("Correlations with the target -> combination = {}:".format(i+1))
    sorted_correlations = sorted(correlations.items(), key=lambda item: item[1], reverse=True)
    print(sorted_correlations[:5])  # Print only the 5 best correlations
    if i == 4:
        best_correlations = sorted_correlations
        best_features = [feature for feature, corr in sorted_correlations[:5]]

print("Selected features for combination 5:", best_features)

# We save the data in a new output csv file
best_features = [feature for feature, corr in sorted_correlations[:5]]
best_features_data = store_new_features[4][best_features]
best_features_data['target'] = y_anemia
best_features_data.to_csv('best_features_data.csv', index=False)

# Use only the best features for the final dataset
X_train, X_test, y_train, y_test = train_test_split(store_new_features[4][best_features], y_anemia, test_size=0.2, random_state=666)

# First, we create the SVM model and the Bayesian Optimizer
model = SVC(probability=True, random_state=123)

# Define the parameter grid for Bayesian Optimization
param_grid = {
    'C': Real(1e-3, 1e+3, prior='log-uniform'),
    'gamma': Real(1e-3, 1e+1, prior='log-uniform'),
    'kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid']),
}

# We define the optimization with cross-validation set to 5
start_time = time.time()
opt = BayesSearchCV(model, param_grid, n_iter=50, cv=5, n_jobs=-1, random_state=42, verbose=3)
opt.fit(X_train, y_train)
optimization_time = time.time() - start_time
print("Bayesian Optimization processed in {} seconds !".format(optimization_time))

# And we print the best parameters
print(f'Best parameters: {opt.best_params_}')
best_model = opt.best_estimator_

# Finally, we evaluate the accuracy of our model
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')

y_pred_optimized = best_model.predict(X_test)
print(f'Optimized Accuracy: {accuracy_score(y_test, y_pred_optimized)}')
print('Optimized Classification Report:')
print(classification_report(y_test, y_pred_optimized))
print('Optimized Confusion Matrix:')
cm_optimized = confusion_matrix(y_test, y_pred_optimized)
sns.heatmap(cm_optimized, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
#plt.savefig("Confusion_Matrix.png", dpi=600, bbox_inches="tight")

# We save the best model and the LabelEncoder
joblib.dump(best_model, 'svm_model.joblib')
joblib.dump(label_encoder_anemia, 'label_encoder_anemia.joblib')

