# DataConsistency: Application 1

In this branch, we will apply our approach on a specific dataset named "Breast Cancer Wisconsin (Diagnostic) Data Set".

## Table of Contents
- [Prerequisites](#prerequisites)
  - [Anaconda and conda environment](#anaconda-and-conda-environment)
  - [Databases](#databases)
- [Goal of the project](#goal-of-the-project)
- [Project architecture](#project-architecture)
  - [1. Load the datasets](#1-load-the-datasets)
  - [2. Pre-treat the data](#2-pre-treat-the-data)
  - [3. Measure the possible correlations](#3-measure-the-possible-correlations)
  - [4. Interpret the results](#4-interpret-the-results)
- [Code and Jupyter notebook available](#code-and-jupyter-notebook-available)

## Prerequisites

### Anaconda and conda environment

It is the same prerequisites as the one specified in the main branch of this project.

### Database

In this project, we use the database named "Breast Cancer Wisconsin (Diagnostic) Data Set", directly available within the scikit-learn library. However, for people who would like to take a look on its content, a version is available on the Kaggle website: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## Goal of the project

***The goal of the project is to show how we can gain both in time and efficency by isolating the best features from the dataset by using the methodology seen in the main branch.*** 

## Project architecture

This project is made of XXX scripts and consists of five main tasks:

[***1. The pre-defined functions***](#-the-pre-defined-functions)
[***1. Isolate the best features without any combinations***](#1-isolate-the-best-features-without-any-combinations)  
[***2. Isolate the best features with combinations***](#2-isolate-the-best-features-with-combinations)  
[***3. Evaluate the training time in these conditions***](#3-evaluate-the-training-time-in-these-conditions)  
[***4. Evaluate the optimization time***](#4-evaluate-the-optimization-time)  

These four scripts will use several functions that we predifined in an external python file called [MyPythonFunctions.py](MyPythonFunctions.py)

Let us see in more details these five aspects.

### 0. The pre-defined functions

In the [MyPythonFunctions.py](MyPythonFunctions.py), we store all the functions we need to define to perform the tasks in the next sections. Among these functions we can find:   
   ***load_breast_cancer_data***: Function to load the dataset "breasr_cancer" from the scikit-learn library   
   ***display_dataset_info***: Function to visualize the dataset   
   ***check_data_loaded***: Function to check if the dataset was correctly loaded   
   ***preprocess_data***: Function to preprocess the data   
   ***create_features***: Function to create interactions between the features and apply the non-linear transformations   
   ***calculate_correlations***: Function to compute the correlations between the new features and the corresponded target   
   ***visualize_top_features***: Function to visualize the best features

### 1. Isolate the best features without any combinations

The script, named as [1-Isolate-best-features-without-combinations](1-Isolate-best-features-without-combinations) starts by loading the dataset:
```python
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

X_breast_cancer_preprocessed = preprocess_data(X_breast_cancer)
```

The model will first load the database of interest. We then print an overview of the dataset:
```python
display_dataset_info(X_breast_cancer, y_breast_cancer, "Breast Cancer Dataset")
```

On your screen, you should then see:
```
Dataset: Breast Cancer Dataset
Features: ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']
Shape: (569, 30)
First 5 rows of the dataset:
   mean radius  mean texture  mean perimeter  mean area  mean smoothness  ...  worst compactness  worst concavity  worst concave points  worst symmetry  worst fractal dimension
0        17.99         10.38          122.80     1001.0          0.11840  ...             0.6656           0.7119                0.2654          0.4601                  0.11890
1        20.57         17.77          132.90     1326.0          0.08474  ...             0.1866           0.2416                0.1860          0.2750                  0.08902
2        19.69         21.25          130.00     1203.0          0.10960  ...             0.4245           0.4504                0.2430          0.3613                  0.08758
3        11.42         20.38           77.58      386.1          0.14250  ...             0.8663           0.6869                0.2575          0.6638                  0.17300
4        20.29         14.34          135.10     1297.0          0.10030  ...             0.2050           0.4000                0.1625          0.2364                  0.07678

[5 rows x 30 columns]
```

Thus, we see that the data are loaded correctly !   

The data are then preprocessed:
```python
X_breast_cancer_preprocessed = preprocess_data(X_breast_cancer)
```

A check is then made to see if the data were loaded correctly:
```python
# 2/ Check the loading of the data
check_data_loaded(X_breast_cancer, y_breast_cancer, "Breast Cancer Dataset")
```

We then establish the features we would like to consider here:
```python
# 3/ Then, create the new features
X_breast_cancer_enhanced = create_features(X_breast_cancer_preprocessed, combination=1) # combination here define the degree of combination we will use
```
As you can see, we set the parameter "combination" to 1, which means that no combinations between the features will be done !   

We then proceed to the calculations of the correlations between the features and the target:
```python
# 4/ Compute the correlations between the new features and the target
correlations_breast_cancer_enhanced = calculate_correlations(X_breast_cancer_enhanced, y_breast_cancer)
top_features_breast_cancer = sorted(correlations_breast_cancer_enhanced.items(), key=lambda item: item[1], reverse=True)[:10]
```
Here, 10 means that we will conserve only the 10 best features !

And finally, we will print the top 10 features with their score, and optionally we can visualize their correlations by plotting them:
```python
# 5/ Print the 10 best features with their score
print("Top 10 features with their correlation score -- Breast Cancer Dataset:")
for feature, score in top_features_breast_cancer:
    print(f"{feature}: {score}")

# 6/ Extract the name of the best features
top_features_names_breast_cancer = [feature for feature, _ in top_features_breast_cancer]

# 7/ Optionnaly, visualize their relationship with the target
#visualize_top_features(X_breast_cancer_enhanced, y_breast_cancer, top_features_names_breast_cancer, "Breast Cancer Dataset Enhanced")
```

On your screen, you should see the following result:
```
Breast Cancer Dataset loaded successfully.
Top 10 features with their correlation score -- Breast Cancer Dataset:
worst concave points: 0.79356601714127
worst perimeter: 0.7829141371737592
mean concave points: 0.7766138400204354
worst radius: 0.7764537785950396
mean perimeter: 0.7426355297258331
worst area: 0.733825034921051
mean radius: 0.7300285113754564
mean area: 0.70898383658539
mean concavity: 0.6963597071719059
worst concavity: 0.659610210369233
```

We can then see that the "worst concave points", "worst perimeter" and "mean concave points" are the three best features that correlate well with the target. 

### 2. Pre-treat the data

Before manipulating them, the data have to be first converted. We first converted the targets into binaries:   

```python
# We convert the target into binaries
from sklearn.preprocessing import LabelEncoder

label_encoder_drug = LabelEncoder()
y_drug = label_encoder_drug.fit_transform(y_drug)

label_encoder_cancer = LabelEncoder()
y_cancer = label_encoder_cancer.fit_transform(y_cancer)
```

and then the features:
```python
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
```

### 3. Measure the possible correlations

The data being pre-processed, we can now start the measure of possible correlations between the features and the corresponding targets. To do this task, we first create new features that are based on combinations between the pre-existing features:

```python
# In order to visualize how the features can correlate well with the target, 
# we create interactions between the features and then we apply non-linear
# tranformations
from sklearn.preprocessing import PolynomialFeatures

def create_features(X):
    poly = PolynomialFeatures(degree=4, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X)
    return pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

X_drug_enhanced = create_features(X_drug_preprocessed)
X_cancer_enhanced = create_features(X_cancer_preprocessed)
```

Importantly, here we decided to use "degree=4" in the "PolynomialFeatures" function. This is purely an empirical choice and the user is free to use either less complex (degree=3) or more complex (degree=5) combinations between the features.   

Once the new features are created and stored in "X_drug_enhanced" and "X_cancer_enhanced", we can use the Pearson correlation method to evaluate the possible correlations between these new features and the corresponding targets:   
```python
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
```

### 4. Interpret the results

Finally we can print the results:
```python
print("Correlations with the target -> drug200.csv:")
print(sorted(correlations_drug_enhanced.items(), key=lambda item: item[1], reverse=True)[:5])  # Print only the 5 best correlations

correlations_cancer_enhanced = calculate_correlations(X_cancer_enhanced, y_cancer)
print("Correlations with the target -> dataset.csv:")
print(sorted(correlations_cancer_enhanced.items(), key=lambda item: item[1], reverse=True)[:5]) # Print only the 5 best correlations
```

and we should observe this on our screen:
```
Correlations with the target -> drug200.csv:
[('Na_to_K', 0.5891198660590571), ('Na_to_K^2', 0.5293393240544266), ('Age Na_to_K^2', 0.47156637415498887), ('BP Na_to_K', 0.46774484303852415), ('Na_to_K^3', 0.45736262390586513)]
Correlations with the target -> dataset.csv:
[('AGE^2 CHRONIC_DISEASE WHEEZING', 0.05834562878524951), ('AGE PEER_PRESSURE WHEEZING CHEST_PAIN', 0.056746749185962), ('AGE CHRONIC_DISEASE WHEEZING CHEST_PAIN', 0.056429262998627674), ('AGE PEER_PRESSURE WHEEZING ALCOHOL_CONSUMING', 0.05616977449562988), ('AGE WHEEZING^2 CHEST_PAIN', 0.05579777889721853)]
```

***How to interpret these results ?***

The results here show the five best correlations for each of the databases. The coefficient is encompassed between -1 and 1, with 0 meaning that no correlations can be found. More information about this method can be found here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

The main remark here is that the features present in drug200.csv show good correlations with their corresponding targets, with the features "Na_to_K", "Age" and "BP" being the most prominent ones. It thus explains why building classification models with these data was feasible in our previous project.   

However, the correlations for features present in lung_cancer.csv show very bad correlations, with the maximum being less than 0.06. It demonstrates that if we try to build an ML model on these data, the prediction would be random as the model will not be able to learn based on these features.   

***What do do next ?***

With these results, the owners of the database [lung_cancer.csv](databases/lung_cancer.csv) should revised their features and create new more relevant features that would be more suitable for next generation of ML-based models in the future. Furthermore, this simple but efficient approach is crucial to predict the potential efficiency of any ML model which would be trained on data. 

## Code and jupyter notebook available

The full code is available here: [data_consistency.py](data_consistency.py).   

The jupyter notebook released on Kaggle is available here: https://www.kaggle.com/code/celerse/data-consistency

If you have any comments, remarks, or questions, do not hesitate to leave a comment or to contact me directly. I would be happy to discuss it directly with you !
