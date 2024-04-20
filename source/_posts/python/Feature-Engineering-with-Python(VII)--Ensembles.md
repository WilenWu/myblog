---
title: 特征工程(VII)--模型集成
tags:
  - Python
  - 机器学习
categories:
  - Python
  - Machine Learning
cover: /img/FeatureEngine.png
top_img: /img/python-top-img.svg
date: 2024-04-20 18:40:52
abbrlink: 425f9947
description:
---

# Ensembles

有时候模型集成可以取得不错的效果。常用的模型集成包括：

- Votting：简单投票或加权平均
- Stacking：简单来说就是学习各个基本模型的预测值来预测最终的结果

我们初步选用 Stacking 集成学习器，采用 LogisticRegression、SVC、GaussianNB、SGDClassifier 、RandomForestClassifier、HistGradientBoostingClassifier作为基分类器。

Jupyter Notebook 代码连接：[machine_learning_ensembles](/ipynb/machine_learning_ensembles)

导入必要的包


```python
import pandas as pd
import numpy as np
import copy
import json
import pickle
import joblib
import lightgbm as lgb
import optuna
import warnings
import gc

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns

# Setting configuration.
pd.set_option('display.float_format', lambda x: '%.5f' %x)
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
```

## 创建数据集


```python
print('Loading data...')
path = '../datasets/Home-Credit-Default-Risk/selected_data.csv'
df = pd.read_csv(path, index_col='SK_ID_CURR')
```

    Loading data...

```python
# Split data into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(
    df.drop(columns="TARGET"), 
    df["TARGET"], 
    test_size=0.25, 
    random_state=SEED
)

print("X_train shape:", X_train.shape)
print('train:', y_train.value_counts(), sep='\n') 
print('valid:', y_valid.value_counts(), sep='\n')
```

    X_train shape: (230633, 835)
    train:
    TARGET
    0    211999
    1     18634
    Name: count, dtype: int64
    valid:
    TARGET
    0    70687
    1     6191
    Name: count, dtype: int64


无序分类（unordered）特征原始编码对于树集成模型（tree-ensemble like XGBoost）是可用的，但对于线性回归模型（like Lasso or LogisticRegression）则必须使用one-hot重编码。因此，我们先把数据重编码。


```python
# Specific feature names and categorical features
feature_name = X_train.columns.tolist()
categorical_feature = X_train.select_dtypes(object).columns.tolist()
```


```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

# Encode categorical features
encoder = make_column_transformer(
    (OneHotEncoder(
        drop='if_binary', 
        min_frequency=0.02, 
        max_categories=20, 
        sparse_output=False,
        handle_unknown='ignore'
    ), categorical_feature),
    remainder='passthrough', 
    verbose_feature_names_out=False,
    verbose=True    
)

print('fitting...')
encoder.fit(X_train)

print('encoding...')
train_dummies = encoder.transform(X_train)
valid_dummies = encoder.transform(X_valid)
print('train data shape:', X_train.shape)
```

    fitting...
    [ColumnTransformer] . (1 of 2) Processing onehotencoder, total=   4.7s
    [ColumnTransformer] ..... (2 of 2) Processing remainder, total=   0.0s
    encoding...
    train data shape: (230633, 835)

```python
del df, X_train, X_valid
gc.collect()
```


    2948

## 创建优化器

先定义一个评估函数


```python
# Define a cross validation strategy
# We use the cross_val_score function of Sklearn. 
# However this function has not a shuffle attribute, we add then one line of code, 
# in order to shuffle the dataset prior to cross-validation

def evaluate(model, X, y, n_folds = 5, verbose=True):
    kf = KFold(n_folds, shuffle=True, random_state=SEED).get_n_splits(X)
    scores = cross_val_score(
        model, 
        X, 
        y, 
        scoring="roc_auc", 
        cv = kf
    )
    if verbose:
        print(f"valid auc: {scores.mean():.3f} +/- {scores.std():.3f}")
    return scores.mean()
```

然后，我们定义一个优化器，对这些基分类器超参数调优。


```python
class Objective:
    estimators = (
  	    LogisticRegression, 
  	    SGDClassifier, 
  	    GaussianNB, 
  	    RandomForestClassifier, 
  	    HistGradientBoostingClassifier
  	)
    def __init__(self, estimator, X, y):
  	    # assert isinstance(estimator, estimators), f"estimator must be one of {estimators}"
        self.model = estimator
        self.X = X
        self.y = y
    
    def __call__(self, trial):
        # Create hyperparameter space
        if isinstance(self.model, LogisticRegression): 
            search_space = dict(
                class_weight = 'balanced', 
                C = trial.suggest_float('C', 0.01, 100.0, log=True),
                l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)  # The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1.
            )
        
        
        elif isinstance(self.model, SGDClassifier): 
            search_space = dict(
                class_weight = 'balanced', 
                loss = trial.suggest_categorical('loss', ['hinge', 'log_loss', 'modified_huber']), 
                alpha = trial.suggest_float('alpha', 1e-5, 10.0, log=True),
                penalty = 'elasticnet',
                l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0),
                early_stopping = True
            )
        
        elif isinstance(self.model, GaussianNB): 
            search_space = dict(
            priors = None
            )
        
        elif isinstance(self.model, RandomForestClassifier): 
            search_space = dict(
                class_weight = 'balanced', 
                n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50),
                max_depth = trial.suggest_int('max_depth', 2, 20),
                max_features = trial.suggest_float('max_features', 0.2, 0.9),
                random_state = SEED
            )
        
        elif isinstance(self.model, HistGradientBoostingClassifier): 
            search_space = dict(
                class_weight = 'balanced', 
                learning_rate = trial.suggest_float('learning_rate', 1e-3, 10.0, log=True),
                max_iter = trial.suggest_int('max_iter', 50, 500, step=50),
                max_depth = trial.suggest_int('max_depth', 2, 20),
                max_features = trial.suggest_float('max_features', 0.2, 0.9),
                l2_regularization = trial.suggest_float('l2_regularization', 1e-3, 10.0, log=True),
                random_state = SEED,
                verbose = 0
            )
    
        # Setting hyperparameters
        self.model.set_params(**search_space) 
    
        # Training with 5-fold CV:
        score = evaluate(self.model, self.X, self.y)
        return score
```

## 超参数优化

并行执行贝叶斯优化


```python
def timer(func):
    import time
    import functools
    def strfdelta(tdelta, fmt):
        hours, remainder = divmod(tdelta, 3600)
        minutes, seconds = divmod(remainder, 60)
        return fmt.format(hours, minutes, seconds)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        click = time.time()
        result = func(*args, **kwargs)
        delta = strfdelta(time.time() - click, "{:.0f} hours {:.0f} minutes {:.0f} seconds")
        print(f"{func.__name__} cost time {delta}")
        return result
    return wrapper

# Creating a pipeline & Hyperparameter tuning

@timer
def tuning(model, X, y):
    # create a study object
    study = optuna.create_study(direction="maximize")
    # Invoke optimization of the objective function.
    objective = Objective(model, X, y)
    study.optimize(
        objective, 
        n_trials = 50,
        timeout = 2400,
        gc_after_trial = True,
        show_progress_bar = True
    )
    print(model, 'best score:', study.best_value) 
    return study
```


```python
Objective.estimators
```


    (sklearn.linear_model._logistic.LogisticRegression,
     sklearn.linear_model._stochastic_gradient.SGDClassifier,
     sklearn.naive_bayes.GaussianNB,
     sklearn.ensemble._forest.RandomForestClassifier,
     sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier)


```python
# opt_results = []
# for model in Objective.estimators:
#     study = tuning(model(), train_dummies, y_train)
#     opt_results.append(study)
#     print(model)
#     print(study.best_trial.params)
```

## 模型训练

集成模型调优


```python
# define the search space and the objecive function
def stacking_obj(trial):
    stacking = StackingClassifier(
        # The `estimators` parameter corresponds to the list of the estimators which are stacked.
        estimators = [
            ('Logit', LogisticRegression(
                class_weight = 'balanced', 
                C = trial.suggest_float('Logit__C', 0.01, 100.0, log=True),
                l1_ratio = trial.suggest_float('Logit__l1_ratio', 0.0, 1.0)  # The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1.
            )),
            ('SGD', SGDClassifier(
                class_weight = 'balanced', 
                loss = trial.suggest_categorical('SGD__loss', ['hinge', 'log_loss', 'modified_huber']), 
                alpha = trial.suggest_float('SGD__alpha', 1e-5, 10.0, log=True),
                penalty = 'elasticnet',
                l1_ratio = trial.suggest_float('SGD__l1_ratio', 0.0, 1.0),
                early_stopping = True
            )),
            ('GaussianNB', GaussianNB())
        ],
        # The final_estimator will use the predictions of the estimators as input
        final_estimator = LogisticRegression(
            class_weight = 'balanced', 
            C = trial.suggest_float('final__C', 0.01, 100.0, log=True),
            # The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1.
            l1_ratio = trial.suggest_float('final__l1_ratio', 0.0, 1.0)  
        ),
        verbose = 1
    )
    score = evaluate(stacking, train_dummies, y_train, n_folds = 3)
    return score
```


```python
# create a study object.
study = optuna.create_study(
    study_name = 'stacking-study',  # Unique identifier of the study.
    direction = 'maximize'
)

# Invoke optimization of the objective function.
study.optimize(
    stacking_obj, 
    n_trials = 100, 
    timeout = 3600,
    gc_after_trial = True,
    show_progress_bar = True
)
```


    valid auc: 0.676 +/- 0.017
    valid auc: 0.669 +/- 0.021
    valid auc: 0.673 +/- 0.016
    valid auc: 0.451 +/- 0.121
    valid auc: 0.592 +/- 0.045
    valid auc: 0.666 +/- 0.017
    valid auc: 0.675 +/- 0.014
    valid auc: 0.666 +/- 0.021
    valid auc: 0.672 +/- 0.016
    valid auc: 0.667 +/- 0.021
    valid auc: 0.672 +/- 0.012

```python
joblib.dump(study, path + "stacking-study.pkl")

study = joblib.load(path + "stacking-study.pkl")

print("Best trial until now:")
print(" Value: ", study.best_trial.value)
print(" Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")
```

    Best trial until now:
     Value:  0.6761396385434888
     Params: 
        Logit__C: 0.020329668727865235
        Logit__l1_ratio: 0.5165207006926232
        SGD__loss: modified_huber
        SGD__alpha: 1.6638099778831132
        SGD__l1_ratio: 0.7330208370976262
        final__C: 14.1468564043383
        final__l1_ratio: 0.4977751012657087

```python
stacking = StackingClassifier(
    # The `estimators` parameter corresponds to the list of the estimators which are stacked.
    estimators = [
        ('Logit', LogisticRegression(
            class_weight = 'balanced', 
            C = 0.020329668727865235,
            l1_ratio = 0.5165207006926232  # The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1.
        )),
        ('SGD', SGDClassifier(
            class_weight = 'balanced', 
            loss = 'modified_huber', 
            alpha = 1.6638099778831132,
            penalty = 'elasticnet',
            l1_ratio = 0.7330208370976262,
            early_stopping = True
        )),
        ('GaussianNB', GaussianNB())
    ],
    # The final_estimator will use the predictions of the estimators as input
    final_estimator = LogisticRegression(
        class_weight = 'balanced', 
        C = 14.1468564043383,
        # The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio = 0.4977751012657087 
    ),
    verbose = 1
)

score = evaluate(stacking, train_dummies, y_train)
```

    valid auc: 0.674 +/- 0.009

```python
stacking.fit(train_dummies, y_train)

train_auc = roc_auc_score(y_train, stacking.predict_proba(train_dummies)[:, 1])
valid_auc = roc_auc_score(y_valid, stacking.predict_proba(valid_dummies)[:, 1])
print('train auc:', train_auc)
print('valid auc:', valid_auc)
```

    train auc: 0.6753919322392181
    valid auc: 0.6752015627178207

