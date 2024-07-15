#!/usr/bin/env python

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score

# We first load the model
best_model = joblib.load('svm_model.joblib')

# And then the LabelEncoder from our previous training
label_encoder_anemia = joblib.load('label_encoder_anemia.joblib')

# We read the new csv file
new_data_path = "expanded_output.csv"
new_data = pd.read_csv(new_data_path)

# We check the data and separate the features from the target
print(new_data.head())

# We store in two variables the features and the target
X_features = new_data.iloc[:, :-1]
y_anemia = new_data.iloc[:, -1]

# First, we convert the target into binaries
y_new = label_encoder_anemia.transform(y_anemia)

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

X_new_preprocessed = preprocess_data(X_features)

# We standardize the new features
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new_preprocessed)

# We create different combinations of new features by creating interactions between the features and then we apply non-linear transformations
def create_features(X, combination, feature_names):
    poly = PolynomialFeatures(degree=combination, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X)
    return pd.DataFrame(X_poly, columns=poly.get_feature_names_out(feature_names))

best_features = ['Hb', 'Sex^2 Hb', 'Number^2 Hb', '%Green pixel', 'Sex^4 Hb']
X_new_features = create_features(X_new_scaled, 5, X_new_preprocessed.columns)
X_new_best_features = X_new_features[best_features]

# We make the predictions
y_pred_new = best_model.predict(X_new_best_features)

# We compute the accuracy and print the results
accuracy = accuracy_score(y_new, y_pred_new)
print(f'Accuracy on the new data: {accuracy}')
print('Predictions:')
print(y_pred_new)
