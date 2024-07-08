# DataConsistency
This introductory project demonstrates the importance of verifying the data consistency prio to use any ML approach.

In this example, we show how to measure possible correlations between the features of the dataset and the corresponded targets. 

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

### Database

In this project, we use two datasets available on Kaggle and stored here in the folder [databases](databases):   
   1. [drug200.csv](databases/drug200.csv): a datasets made of 200 lines, where depending onn several features a specific drug is assigned (A/B/C/X/Y). More informations are availble on the folling website: https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees/data. Furthermore, we also recently dedicated a ML project which is now released on GitHub at this adress: https://github.com/Fredericcelerse/DrugClassification/tree/main   
   2. [lung_cancer.csv](databases/lung_cancer.csv): This new database, released few weeks ago on Kaggle (https://www.kaggle.com/datasets/akashnath29/lung-cancer-dataset) consists of 3000 lines, where depending on the features we say if YES or NO the patient has a lung cancer.

## Initial goal of the project

The initial goal of this project was to build a ML model, very similar to what we made before for the "DrugClassification" project (https://github.com/Fredericcelerse/DrugClassification/tree/main), and where a patient could  evaluate the risk to have a lung cancer based on his symptomas.

However, every models we used only provided accuracy not higher than 54.5%, showing very randomicity in our predictions. On Kaggle, similar projects also demonstrate the same accuracy by employing several other methods. It finally shows that the data cannot be used accurately to build a ML model.

But instead of just concluding on that, I would like to show ***why it is so problematic in that case to build a ML model***. This is why we dedicate this project in building a small but efficient pipeline which can afford simply explanations on this issue.

## Project architecture

This project is made of one script labeled [data_consistency.py](data_consistency.py) and consists of four main tasks:

***1. Load the datasets***   
***2. Pre-treat the data***   
***3. Measure the possible correlations***   
***4. Interpret the results***

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



### 4. Interpret the results


