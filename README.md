# DataConsistency: Application 2

In this branch, we will apply our approach on a specific dataset named "Anaemia Prediction Dataset".

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

In this project, we use the database named "BAnaemia Prediction Dataset", available on the Kaggle website: [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/humairmunir/anaemia-prediction-dataset/data)

## Goal of the project

***The goal of the project is to apply our approach of combining features in order to evaluate the efficiency of future ML models built on them*** 

## Project architecture

This project is made of 2 scripts and consists of three main tasks:

[***1. Load the data and isolate the best features with combinations***](#1-load-the-data-and-isolate-the-best-features-with-combinations)  
[***2. Train and evaluate an SVM model with Bayesian Optimization***](#2-train-and-evaluate-an-svm-model-with-bayesian-optimization) 
[***3. Save the model and apply it on external data***](#3-save-the-model-and-apply-it-on-external-data)  

Let us see in more details these aspects.

### 1. Load the data and isolate the best features with combinations

The methodology is very standard to the one we are using here for isolating relevant features:
- We first load the data
- Then we preprocess them using python functions
- We build new features based on combinations between the initial features
- We evaluate their correlation with the respective targets using the Pearson coefficient
- Finally, we conserve the 5 best features

To do this task, you can execute in the terminal the code:
```
python 1-Create-Features-and-Train-Model
```

As a result, you will obtain:
```
   Number Sex  %Red Pixel  %Green pixel  %Blue pixel        Hb Anaemic
0       1   M   43.170845     30.945626    25.921971  6.252659     Yes
1       2   F   43.163481     30.306974    26.759843  8.578865     Yes
2       3   F   46.269997     27.315656    26.028556  9.640936     Yes
3       4   F   45.054787     30.469816    24.460797  4.794217     Yes
4       5  M    45.061884     31.218572    24.071714  8.865329     Yes
New combined features with combination = 1 created !
New combined features with combination = 2 created !
New combined features with combination = 3 created !
New combined features with combination = 4 created !
New combined features with combination = 5 created !
Correlations with the target -> combination = 1:
[('Hb', 0.839850957687261), ('%Green pixel', 0.6308513695257353), ('%Red Pixel', 0.4016312389715565), ('Number', 0.3818279000835077), ('Sex', 0.1627221488089796)]
Correlations with the target -> combination = 2:
[('Hb', 0.839850957687261), ('%Green pixel', 0.6308513695257353), ('Number^2', 0.5475906890322616), ('%Red Pixel', 0.4016312389715565), ('Number', 0.3818279000835077)]
Correlations with the target -> combination = 3:
[('Hb', 0.839850957687261), ('Sex^2 Hb', 0.6682132367773528), ('Number^2 Hb', 0.6576444881621494), ('%Green pixel', 0.6308513695257353), ('Number^2', 0.5475906890322616)]
Correlations with the target -> combination = 4:
[('Hb', 0.839850957687261), ('Sex^2 Hb', 0.6682132367773528), ('Number^2 Hb', 0.6576444881621494), ('%Green pixel', 0.6308513695257353), ('Number^2', 0.5475906890322616)]
Correlations with the target -> combination = 5:
[('Hb', 0.839850957687261), ('Sex^2 Hb', 0.6682132367773528), ('Number^2 Hb', 0.6576444881621494), ('%Green pixel', 0.6308513695257353), ('Sex^4 Hb', 0.6093366334419581)]
Selected features for combination 5: ['Hb', 'Sex^2 Hb', 'Number^2 Hb', '%Green pixel', 'Sex^4 Hb']
```
We can then see that the combination 5 offers the best results, we thus decide to conserve these features: 'Hb', 'Sex^2 Hb', 'Number^2 Hb', '%Green pixel', 'Sex^4 Hb'

### 2. Train and evaluate an SVM model with Bayesian Optimization

In the continuity, the code will select these features and build an SVM model and autoomatically optimized the hyperparameters using the Bayesian Optimization approach. As a result, you should see in your console:
```
Fitting 5 folds for each of 1 candidates, totalling 5 fits
[CV 4/5] END C=0.28881766539144715, gamma=0.8145222883402798, kernel=sigmoid;, score=0.988 total time=   0.0s
[CV 1/5] END C=0.28881766539144715, gamma=0.8145222883402798, kernel=sigmoid;, score=0.975 total time=   0.0s
[CV 3/5] END C=0.28881766539144715, gamma=0.8145222883402798, kernel=sigmoid;, score=0.988 total time=   0.0s
[CV 2/5] END C=0.28881766539144715, gamma=0.8145222883402798, kernel=sigmoid;, score=1.000 total time=   0.0s
[CV 5/5] END C=0.28881766539144715, gamma=0.8145222883402798, kernel=sigmoid;, score=0.975 total time=   0.0s
Fitting 5 folds for each of 1 candidates, totalling 5 fits
[CV 1/5] END C=105.76211650904162, gamma=3.413981076557398, kernel=poly;, score=1.000 total time=   0.0s
[CV 2/5] END C=105.76211650904162, gamma=3.413981076557398, kernel=poly;, score=1.000 total time=   0.0s
[CV 4/5] END C=105.76211650904162, gamma=3.413981076557398, kernel=poly;, score=1.000 total time=   0.0s
[CV 3/5] END C=105.76211650904162, gamma=3.413981076557398, kernel=poly;, score=1.000 total time=   0.0s
[CV 5/5] END C=105.76211650904162, gamma=3.413981076557398, kernel=poly;, score=0.988 total time=   0.0s
...
...
...
Fitting 5 folds for each of 1 candidates, totalling 5 fits
[CV 1/5] END C=542.8146138331425, gamma=0.018850745969253083, kernel=sigmoid;, score=0.988 total time=   0.0s
[CV 2/5] END C=542.8146138331425, gamma=0.018850745969253083, kernel=sigmoid;, score=1.000 total time=   0.0s
[CV 3/5] END C=542.8146138331425, gamma=0.018850745969253083, kernel=sigmoid;, score=0.988 total time=   0.0s
[CV 4/5] END C=542.8146138331425, gamma=0.018850745969253083, kernel=sigmoid;, score=1.000 total time=   0.0s
[CV 5/5] END C=542.8146138331425, gamma=0.018850745969253083, kernel=sigmoid;, score=0.988 total time=   0.0s
Bayesian Optimization processed in 61.32900285720825 seconds !
Best parameters: OrderedDict([('C', 68.82665659460292), ('gamma', 0.001), ('kernel', 'linear')])
Cross-validation scores: [1. 1. 1. 1. 1.]
Mean cross-validation score: 1.0
Optimized Accuracy: 1.0
Optimized Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        51
           1       1.00      1.00      1.00        49

    accuracy                           1.00       100
   macro avg       1.00      1.00      1.00       100
weighted avg       1.00      1.00      1.00       100

Optimized Confusion Matrix:
```

And  a picture of the confusion matrix, which should show you a perfect match between the predictions and the references ! 

### 3. Save the model and apply it on external data

The model and the encoder (needed in the preprocessing of the data) are automatically saved in two external file, named [svm_model.joblib](svm_model.joblib) and [label_encoder_anemia.joblib](label_encoder_anemia.joblib). The aim now is to be able to use our model to check if our value is well trained without any extrapolation (as it could be suggested with such a nice score ...). To do that, we have a script labeled [2-Test-the-Model.py](2-Test-the-Model.py) that we can launch in our terminal:
```
python 2-Test-the-Model.py
```

As a result, you should observe this in your terminal:

```
   Number Sex  %Red Pixel  %Green pixel  %Blue pixel        Hb Anaemic
0       1   M   43.264176     30.838924    25.899587  6.297293     Yes
1       2   F   43.144832     30.171404    26.692997  8.608315     Yes
2       3   F   46.506491     27.430905    26.051133  9.713010     Yes
3       4   F   44.963982     30.519205    24.499161  4.809385     Yes
4       5  M    45.069466     31.089378    23.853518  8.995228     Yes
Accuracy on the new data: 1.0
Predictions:
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
```

Finally, we can see a perfect prediction, meaning that:
- First, the features are perfectly consistent with their respective target
- Then, the hyperparameters are well tuned
- And finally, no extrapolations are observed, meaning that our model can be used safely !

As an opening, if there is doubt in certain prediction, the option "probability=True" can be used to evaluate its uncertainty. To do that, you can change these lines in the code [2-Test-the-Model.py](2-Test-the-Model.py):
```
> y_pred_new = best_model.predict(X_new_best_features)
< y_pred_proba = best_model.predict_proba(X_new_best_features)
< import numpy as np
< y_pred_new = np.argmax(y_pred_new, axis=1)
```

and 

```
< print(f"Probability: {y_pred_proba[0]}")
```

These slitghly modifications will give you access to the probability of the prediction  !
