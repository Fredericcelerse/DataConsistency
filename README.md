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
   ***train_RF***: Function to train a Random Forest classification algorithm    
   ***optimize_model***: Function to automatized the hyperparameters tuning for RF
   ***train_and_evaluate_model***: Function to retrain and evaluate RF model with the best parameters

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
```

We then build the RF models and launch the different training and measured the time. 

```python
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
```

and we do the same for the other models:

```python
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
```

You should observe on your screen the following results:
```
Breast Cancer Dataset loaded successfully.
1/10 done !
2/10 done !
3/10 done !
4/10 done !
5/10 done !
6/10 done !
7/10 done !
8/10 done !
9/10 done !
10/10 done !

CASE1: THE BEST NON_COMBINED FEATURES

Number of features: 1 --- Training time: 0.18052911758422852 --- Accuracy: 0.9035087719298246
Number of features: 2 --- Training time: 0.1452639102935791 --- Accuracy: 0.9385964912280702
Number of features: 3 --- Training time: 0.1372699737548828 --- Accuracy: 0.9385964912280702
Number of features: 4 --- Training time: 0.15772390365600586 --- Accuracy: 0.956140350877193
Number of features: 5 --- Training time: 0.15401101112365723 --- Accuracy: 0.956140350877193
Number of features: 6 --- Training time: 0.15012788772583008 --- Accuracy: 0.956140350877193
Number of features: 7 --- Training time: 0.14864802360534668 --- Accuracy: 0.956140350877193
Number of features: 8 --- Training time: 0.14806079864501953 --- Accuracy: 0.9649122807017544
Number of features: 9 --- Training time: 0.16820979118347168 --- Accuracy: 0.956140350877193
Number of features: 10 --- Training time: 0.16501927375793457 --- Accuracy: 0.956140350877193

CASE2: THE BEST NON_COMBINED FEATURES

Number of features: 569 --- Training time: 0.207841157913208 --- Accuracy: 0.9649122807017544
1/10 done !
2/10 done !
3/10 done !
4/10 done !
5/10 done !
6/10 done !
7/10 done !
8/10 done !
9/10 done !
10/10 done !

CASE3: THE BEST NON_COMBINED FEATURES

Number of features: 1 --- Training time: 0.13260984420776367 --- Accuracy: 0.9385964912280702
Number of features: 2 --- Training time: 0.13120627403259277 --- Accuracy: 0.9649122807017544
Number of features: 3 --- Training time: 0.13457822799682617 --- Accuracy: 0.9736842105263158
Number of features: 4 --- Training time: 0.1491100788116455 --- Accuracy: 0.9736842105263158
Number of features: 5 --- Training time: 0.14768600463867188 --- Accuracy: 0.956140350877193
Number of features: 6 --- Training time: 0.1454331874847412 --- Accuracy: 0.956140350877193
Number of features: 7 --- Training time: 0.15279293060302734 --- Accuracy: 0.956140350877193
Number of features: 8 --- Training time: 0.14442920684814453 --- Accuracy: 0.9649122807017544
Number of features: 9 --- Training time: 0.16461610794067383 --- Accuracy: 0.9649122807017544
Number of features: 10 --- Training time: 0.15948224067687988 --- Accuracy: 0.9649122807017544

CASE4: THE BEST NON_COMBINED FEATURES

Number of features: 569 --- Training time: 3.3202831745147705 --- Accuracy: 0.9824561403508771
```

Finally, we can see the following things:
- First, taking into account our combining features lead to slightly better results
- Taking all the combined features (569) here lead to a remarkable result (98% of accuracy Vs 96% with the initial ones)
- However, the training time is much more higher (3.3 s Vs 0.15 s in average), which could be an issue for larger database

Now we saw that one of the best result could be obtained with our 4 best combined features, let's see how it could be used to optimize the simulation time related to the hyperparameters optimization process. 

### 5. Evaluate the optimization time

Finally, we see here how the number of features is influencing the simulation we need for the hyperparameters tuning. We use the script [3-Optimized-RF.py](3-Optimized-RF.py), which will make the optimization with the 4 best features and the whole parameters.   

In the terminal, you can launch the code by typing:
```bash
python 3-Optimized-RF.py
```

And as a result, you will obtain:

```
Best parameters for the RF model with only the 4 best features:
OrderedDict([('max_depth', 11), ('min_samples_leaf', 1), ('min_samples_split', 9), ('n_estimators', 220)])
Optimization time: 95.3313 seconds

Optimized model with the 4 best features:
Accuracy: 0.9737
Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.98      0.97        43
           1       0.99      0.97      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114

ROC AUC Score: 0.9948



```

***How to interpret these results ?***

- First, we can say that we do not see any improvement in our results. This could be due to two main reasons:
  - First, the results are already very impressive, and it is thus hard to improve the model
  - But also, we maybe need to increase our research space defined by our Bayesian Optimization, but a the price of more computational time ...
- Then, the optimization time using only the 4 best features (96 s) is much more lower than the one needed with the full features (), but for only a small piece of improvement. In our case, it does not matter because of it is fast, but it could be rapidly a bigger issue for larger and more complex databases in future case studies ...
- Combining features together does not take too much time, and give you a nice overview of the quality of the data you have.

***What do do next ?***

Apply this approach of the best combined features to tackle more complex and bigger databases in different type of fields, including life science or finance for instance! 

## Code and jupyter notebook available

The jupyter notebook released on Kaggle is available here: XXX

If you have any comments, remarks, or questions, do not hesitate to leave a comment or to contact me directly. I would be happy to discuss it directly with you !
