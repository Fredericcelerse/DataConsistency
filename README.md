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

This project is made of 5 scripts and consists of five main tasks:

[***1. The pre-defined functions***](#-the-pre-defined-functions)   
[***2. Isolate the best features without any combinations***](#1-isolate-the-best-features-without-any-combinations)  
[***3. Isolate the best features with combinations***](#2-isolate-the-best-features-with-combinations)  
[***4. Evaluate the training time in these conditions***](#3-evaluate-the-training-time-in-these-conditions)  
[***5. Evaluate the optimization time***](#4-evaluate-the-optimization-time)  

These four scripts will use several functions that we predifined in an external python file called [MyPythonFunctions.py](MyPythonFunctions.py)

Let us see in more details these five aspects.

### 1. The pre-defined functions

In the [MyPythonFunctions.py](MyPythonFunctions.py), we store all the functions we need to define to perform the tasks in the next sections. Among these functions we can find:   
   ***load_breast_cancer_data***: Function to load the dataset "breasr_cancer" from the scikit-learn library   
   ***display_dataset_info***: Function to visualize the dataset   
   ***check_data_loaded***: Function to check if the dataset was correctly loaded   
   ***preprocess_data***: Function to preprocess the data   
   ***create_features***: Function to create interactions between the features and apply the non-linear transformations   
   ***calculate_correlations***: Function to compute the correlations between the new features and the corresponded target   
   ***visualize_top_features***: Function to visualize the best features

### 2. Isolate the best features without any combinations

The script, named as [1-Isolate-best-features](1-Isolate-best-features) starts by loading the dataset:
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

The code can be launched by entering in your terminal:
```bash
python 1-Isolate-best-features
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

We can then see that the "worst concave points", "worst perimeter" and "mean concave points" are the three best features that correlate well with the target, with the best result at 0.79. 

### 3. Isolate the best features with combinations

Now, we will see how combining the data together can help to increase the accuracy of the model. To do that, we will use the same script as before, but by changing only one line:
```python
# 3/ Then, create the new features
X_breast_cancer_enhanced = create_features(X_breast_cancer_preprocessed, combination=4) # combination here define the degree of combination we will use
```
Here, we will select 4, which means that we will combine 4 features together in each of the new features we will create. As a total, we will thus create 4638 new features !

As before, you can now launch the code in your terminal:
```bash
python 1-Isolate-best-features
```

and you should now observe in your terminal the following results:
```
Breast Cancer Dataset loaded successfully.
Top 10 features with their correlation score -- Breast Cancer Dataset:
worst perimeter worst smoothness: 0.8077304593739741
worst radius worst smoothness: 0.8066751990639625
worst texture worst concave points: 0.8061003882272
mean texture worst concave points: 0.8038085065816557
worst radius worst concave points: 0.8004479144662467
mean radius worst texture worst concave points: 0.8002599353725561
mean radius worst concave points: 0.79787539682551
mean concave points worst texture: 0.7974094767316239
mean perimeter worst texture worst concave points: 0.7967988135459076
mean perimeter worst smoothness: 0.7965052344416643
```

You can observe here a better correlation between the new features and the target, with a maximal score of 0.807. It thus shows the importance of considering the combination of features together, as we did previously in the main branch.   

Importantly, the choice to set the combination to 4 is purely empiric. However, this parameter can be tuned in future studies, but empirically, I almost observed good results when this parameter was set between 3 and 6. 

### 4. Evaluate the training time

Once we captured the best features (with and without combinations), it is interesting to see how it impacts the training of an ML model. In our case, we select the Random Forest (RF) model, but of course other type of classification models could be used in further studies. Here, we will focus to capture the evolution of the two main parameters, which are:
- The training time as a function of the number of features
- The accuracy as a function of the number of features

To do this task, we will consider 3 subtasks:
- An RF model trained only with the 10 best non-combined features
- An RF model trained with the whole non-combined features
- An RF model trained with the 10 best combined features
- An RF model trained with the whole combined features

To do this task, we resort to the script [2-Trained-RF-model.py](2-Trained-RF-model.py).   

We first load the dataset, preprocess the data and create the combined features. 
```python
#!/usr/bin/env python

import pandas as pd
import time

# We import here our mad-home libraries
from MyPythonFunctions import (
    load_breast_cancer_data,
    display_dataset_info,
    check_data_loaded,
    preprocess_data,
    create_features
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
   ' worst radius',
    'mean perimeter',
    'worst area',
    'mean radius',
    'mean area',
    'mean concavity',
    'worst concavity'
]

# 3/ And now the top 10 features we identified previously with combinations
top_combined_features_names_breast_cancer = [
    worst perimeter worst smoothness',
    worst radius worst smoothness',
    worst texture worst concave points',
    mean texture worst concave points',
    worst radius worst concave points',
    mean radius worst texture worst concave points',
    mean radius worst concave points',
    mean concave points worst texture',
    mean perimeter worst texture worst concave points',
    mean perimeter worst smoothness'
]

# 4/ We create the same new features as before
X_breast_cancer_enhanced = create_features(X_breast_cancer_preprocessed)

# 5/ We extract the best features
X_breast_cancer_selected1 = X_breast_cancer_enhanced[top_features_names_breast_cancer]
X_breast_cancer_selected2 = X_breast_cancer_enhanced[top_combined_features_names_breast_cancer]
```

We then build the RF models and launch the different training and measured the time. 

### 5. Evaluate the optimization time



***How to interpret these results ?***


***What do do next ?***

 

## Code and jupyter notebook available

The full code is available here:    

The jupyter notebook released on Kaggle is available here: 

If you have any comments, remarks, or questions, do not hesitate to leave a comment or to contact me directly. I would be happy to discuss it directly with you !
