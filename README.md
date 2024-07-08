# DataConsistency
![Alt text](Data_consistency_in_AI.png)

This introductory project demonstrates the importance of verifying data consistency prior to using any ML approach.   

In this example, we show how to measure possible correlations between the features of the dataset and the corresponding targets.

## Table of Contents
- [Prerequisites](#prerequisites)
  - [Anaconda](#anaconda)
  - [Setup conda environment](#setup-conda-environment)
  - [Databases](#databases)
- [Initial goal of the project](#initial-goal-of-the-project)
- [Project architecture](#project-architecture)
  - [1. Load the datasets](#1-load-the-datasets)
  - [2. Pre-treat the data](#2-pre-treat-the-data)
  - [3. Measure the possible correlations](#3-measure-the-possible-correlations)
  - [4. Interpret the results](#4-interpret-the-results)
- [Code and Jupyter notebook available](#code-and-jupyter-notebook-available)

## Prerequisites

### Anaconda

To execute the code, we will set up a specific environment using Anaconda. To install it, visit [Anaconda Installation](https://docs.anaconda.com/free/anaconda/install/).

### Setup conda environment

First, create the conda environment:
```
conda create -n DataConsistency python=3.8
```

Then, activate the conda environment:
```
conda activate DataConsistency
```

Once the environment is properly created, install the necessary Python libraries to execute the code:
```
pip install scikit-learn numpy pandas scipy
```

### Databases

In this project, we use two databases available on Kaggle and stored here in the folder [databases](databases):   
   1. [drug200.csv](databases/drug200.csv): a dataset made of 200 lines, where depending on several features a specific drug is assigned (A/B/C/X/Y). More information is available on the following website: https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees/data. Furthermore, we also recently dedicated an ML project which is now released on GitHub at this address: https://github.com/Fredericcelerse/DrugClassification/tree/main    
   2. [lung_cancer.csv](databases/lung_cancer.csv): This new database, released a few weeks ago on Kaggle (https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset) consists of 3000 lines, where depending on the features we say if YES or NO the patient has lung cancer.   

## Initial goal of the project

The initial goal of this project was to build an ML model, very similar to what we did before for the "DrugClassification" project (https://github.com/Fredericcelerse/DrugClassification/tree/main), where a patient could evaluate the risk of having lung cancer based on his symptoms.   

However, every model we used only provided accuracy not higher than 54.5%, showing very random results in our predictions. On Kaggle, similar projects also demonstrate the same accuracy by employing several other methods. It finally shows that the data cannot be used accurately to build an ML model.   

But instead of just concluding on that, I would like to show ***why it is so problematic in this case to build an ML model***. This is why we dedicate this project to building a small but efficient pipeline which can provide simple explanations on this issue.

## Project architecture

This project is made of one script labeled [data_consistency.py](data_consistency.py) and consists of four main tasks:

[***1. Load the datasets***](#1-load-the-datasets)  
[***2. Pre-treat the data***](#2-pre-treat-the-data)  
[***3. Measure the possible correlations***](#3-measure-the-possible-correlations)  
[***4. Interpret the results***](#4-interpret-the-results)  

Let us see in more details these four aspects

### 1. Load the datasets

The script starts by loading the two datasets:
```python
#!/usr/bin/env python

# We first read the data
import pandas as pd
drug_data_path = "/kaggle/input/drugs-a-b-c-x-y-for-decision-trees/drug200.csv"
cancer_data_path = "/kaggle/input/lung-cancer-dataset/dataset.csv"

drug_data = pd.read_csv(drug_data_path)
cancer_data = pd.read_csv(cancer_data_path)
```
The model will first load the two databases of interest. We then print an overview of these two datasets in order to check if everything was loaded correctly:
```python
# We check the data and separate the features to the target

print(drug_data.head())
print(cancer_data.head())

X_drug = drug_data.iloc[:, :-1]
y_drug = drug_data.iloc[:, -1]

X_cancer = cancer_data.iloc[:, :-1]
y_cancer = cancer_data.iloc[:, -1]
```

On your screen, you should then see:
```
   Age Sex      BP Cholesterol  Na_to_K   Drug
0   23   F    HIGH        HIGH   25.355  drugY
1   47   M     LOW        HIGH   13.093  drugC
2   47   M     LOW        HIGH   10.114  drugC
3   28   F  NORMAL        HIGH    7.798  drugX
4   61   F     LOW        HIGH   18.043  drugY
  GENDER  AGE  SMOKING  YELLOW_FINGERS  ANXIETY  PEER_PRESSURE  \
0      M   65        1               1        1              2   
1      F   55        1               2        2              1   
2      F   78        2               2        1              1   
3      M   60        2               1        1              1   
4      F   80        1               1        2              1   

   CHRONIC_DISEASE  FATIGUE  ALLERGY  WHEEZING  ALCOHOL_CONSUMING  COUGHING  \
0                2        1        2         2                  2         2   
1                1        2        2         2                  1         1   
2                1        2        1         2                  1         1   
3                2        1        2         1                  1         2   
4                1        2        1         2                  1         1   

   SHORTNESS_OF_BREATH  SWALLOWING_DIFFICULTY  CHEST_PAIN LUNG_CANCER  
0                    2                      2           1          NO  
1                    1                      2           2          NO
2                    2                      1           1         YES  
3                    1                      2           2         YES  
4                    1                      1           2          NO  
```

Thus, we see that the data are loaded correctly !

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
