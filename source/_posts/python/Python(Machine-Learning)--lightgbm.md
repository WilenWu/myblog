---
title: Python手册(Machine Learning)--LightGBM
tags:
  - Python
  - 机器学习
categories:
  - Python
  - Machine Learning
cover: /img/LightGBM_cover.svg
top_img: /img/LightGBM_cover.svg
abbrlink: '44910830'
date: 2024-01-25 22:15:00
description:
---

# Overview

```python
import lightgbm as lgb
```

## Data Structure

| 数据结构                                                     | 说明                   |
| ------------------------------------------------------------ | ---------------------- |
| `lgb.Dataset`(data)                                          | LightGBM数据集         |
| [`lgb.Booster`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html#lightgbm.Booster) | LightGBM中的模型       |
| `lgb.CVBooster`                                              | CVBooster in LightGBM. |

## Simple example

**Step 1:**  Load the dataset

```python
# load or create your dataset
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create train dataset for lightgbm
dtrain = lgb.Dataset(X_train, y_train)
# In LightGBM, the validation data should be aligned with training data.
deval = lgb.Dataset(X_test, y_test, reference=dtrain)

# if you want to re-use data, remember to set free_raw_data=False
dtrain = lgb.Dataset(X_train, y_train, free_raw_data=False)
```

LightGBM 可以直接使用分类特征，而不需要 one-hot 编码，且比编码后的更快 (about 8x speed-up)

```python
# Specific feature names and categorical features
```

> Note: 在构建 Dataset 前，先把分类特征转换成整数型

**Step 2: Setting Parameters**

```python
# LightGBM can use a dictionary to set Parameters.

# Booster parameters:
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
params['metric'] = 'l2'

# You can also specify multiple eval metrics:
params['metric'] = ['l2', 'l1']
```

**Step 3: Training**

```python
# Training a model requires a parameter list and data set:
bst = lgb.train(params,
                dtrain,
                num_boost_round=20,
                valid_sets=deval,
                callbacks=[lgb.early_stopping(stopping_rounds=5)])

# Training with 5-fold CV:
lgb.cv(params, dtrain, num_boost_round=20, nfold=5)
```

**Step 4: Save and load model**

```python
# Save model to file:
bst.save_model('model.txt')  
bst = lgb.Booster(model_file='model.txt')
# can only predict with the best iteration (or the saving iteration)

# Dump model to JSON:
import json
model_json = bst.dump_model()
with open('model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)

# Dump model with pickle
import pickle
with open('model.pkl', 'wb') as fout:
    pickle.dump(gbm, fout)
with open('model.pkl', 'rb') as fin:
    pkl_bst = pickle.load(fin)
# can predict with any iteration when loaded in pickle way
```

**Step 5:  Predict**

```python
# A model that has been trained or loaded can perform predictions on datasets:
y_pred = bst.predict(X_test)

# If early stopping is enabled during training, you can get predictions from the best iteration with bst.best_iteration:
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
```

**Step 6:  Evaluating**

```python
from sklearn.metric import mean_squared_error
rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
print(f'The RMSE of prediction is: {rmse_test}')
```


# Parameters

## Booster parameters

learning_rate = 0.1
num_leaves = 255
num_trees = 500
num_threads = 16
min_data_in_leaf = 0
min_sum_hessian_in_leaf = 100

lightgbm.train(params, train_set, num_boost_round=100, valid_sets=None, valid_names=None, feval=None, init_model=None, feature_name='auto', categorical_feature='auto', keep_training_booster=False, callbacks=None)

## Training



## Callbacks

| 方法                                      | Create a callback             |
| ----------------------------------------- | ----------------------------- |
| `lgb.early_stopping(stopping_rounds)`     | 传递提前停止策略              |
| `lgb.log_evaluation([period, show_stdv])` | 输出评估结果的频率            |
| `lgb.record_evaluation(eval_result)`      | 在`eval_result`中记录评估结果 |
| `lgb.reset_parameter(**kwargs)`           | 第一次迭代后重置参数          |

```python
evals_result = {}  # to record eval results for plotting
bst = lgb.train(params, 
                dtrain, 
                num_boost_round=20, 
                valid_sets=deval, 
                callbacks=[lgb.early_stopping(stopping_rounds=5)]
               )
bst.save_model('model.txt', num_iteration=bst.best_iteration)
```

## Self-define

```python
# self-defined objective function
# f(preds: array, train_data: Dataset) -> grad: array, hess: array
# log likelihood loss
def loglikelihood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess

# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: str, eval_result: float, is_higher_better: bool
# binary error
# NOTE: when you do customized loss function, the default prediction value is margin
# This may make built-in evaluation metric calculate wrong results
# For example, we are doing log likelihood loss, the prediction is score before logistic transformation
# Keep this in mind when you use the customization
def binary_error(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return 'error', np.mean(labels != (preds > 0.5)), False

# Pass custom objective function through params
params_custom_obj = copy.deepcopy(params)
params_custom_obj['objective'] = loglikelihood

bst = lgb.train(params_custom_obj,
                lgb_train,
                num_boost_round=10,
                feval=binary_error,
                valid_sets=lgb_eval)

# another self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: str, eval_result: float, is_higher_better: bool
# accuracy
# NOTE: when you do customized loss function, the default prediction value is margin
# This may make built-in evaluation metric calculate wrong results
# For example, we are doing log likelihood loss, the prediction is score before logistic transformation
# Keep this in mind when you use the customization
def accuracy(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return 'accuracy', np.mean(labels == (preds > 0.5)), True
 # Pass custom objective function through params
params_custom_obj = copy.deepcopy(params)
params_custom_obj['objective'] = loglikelihood

bst = lgb.train(params_custom_obj,
                lgb_train,
                num_boost_round=10,
                feval=[binary_error, accuracy],
                valid_sets=lgb_eval)
```

# Plotting

| [`plot_importance`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.plot_importance.html#lightgbm.plot_importance)(booster[, ax, height, xlim, ...]) | Plot model's feature importances.                            |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`plot_split_value_histogram`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.plot_split_value_histogram.html#lightgbm.plot_split_value_histogram)(booster, feature) | Plot split value histogram for the specified feature of the model. |
| [`plot_metric`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.plot_metric.html#lightgbm.plot_metric)(booster[, metric, ...]) | Plot one metric during training.                             |
| [`plot_tree`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.plot_tree.html#lightgbm.plot_tree)(booster[, ax, tree_index, ...]) | Plot specified tree.                                         |
| [`create_tree_digraph`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.create_tree_digraph.html#lightgbm.create_tree_digraph)(booster[, tree_index, ...]) | Create a digraph representation of specified tree.           |

plot_importance(booster[, ax, height, xlim, ...])

Plot model's feature importances.

plot_split_value_histogram(booster, feature)

Plot split value histogram for the specified feature of the model.

plot_metric(booster[, metric, ...])

Plot one metric during training.

plot_tree(booster[, ax, tree_index, ...])

Plot specified tree.

create_tree_digraph(booster[, tree_index, ...])

Create a digraph representation of specified tree.

```python
evals_result = {}  # to record eval results for plotting
gbm = lgb.train(
    params,
    dtrain,
    num_boost_round=100,
    valid_sets=[dtrain, dtest],
    callbacks=[
        lgb.log_evaluation(10),
        lgb.record_evaluation(evals_result)
    ]
)

# Plotting metrics recorded during training
ax = lgb.plot_metric(evals_result, metric='l1')
plt.show()

# Plotting feature importances
ax = lgb.plot_importance(gbm, max_num_features=10)
plt.show()

# Plotting split value histogram
ax = lgb.plot_split_value_histogram(gbm, feature='f26', bins='auto')
plt.show()

# Plotting 54th tree (one tree use categorical feature to split)
ax = lgb.plot_tree(gbm, tree_index=53, figsize=(15, 15), show_info=['split_gain'])
plt.show()

# Plotting 54th tree with graphviz
graph = lgb.create_tree_digraph(gbm, tree_index=53, name='Tree54')
graph.render(view=True)
```

# Continue training

```python
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    init_model=gbm,
                    feature_name=x_cols,
                    early_stopping_rounds=10,
                    verbose_eval=False,
                    keep_training_booster=True) # 增量训练
```

lightGBM有两种增量的方式训练，一种是增加树的方式，一种是更新叶子节点权重的方式

方案1. lgb.train(init_model,keep_training_booster=True) https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html#lightgbm.train
		init_model=clf,
方案2. 调用refit方法 https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html#lightgbm.Booster.refit

```python
# init_model accepts:
# 1. model file name
# 2. Booster()
bst = lgb.train(params,
                dtrain,
                num_boost_round=10,
                init_model=bst, # or 'model.txt'
                valid_sets=deval)
```

如果gbm不为None，那么就是在上次的基础上添加 num_boost_round 棵新树



```python
reader = pd.read_csv('./data cleaned.csv', chunksize=500000) 
params = {
	'task':'refit', 
	'refit_decay_rate': 0.9,
	'boosting_type':'gbdt',
	'objective':'binary',
	'metric':'auc'
	}

for i, chunk in enumerate(reader):
	X_train, y_train = chunk.drop('label', axis=1), chunk['label'] 
	dtrain=lgb.Dataset(X_train, y_train, free_raw_data=False) 
	print('--'*15 + f'chunk {i+1}, size = {len(y_train)}' + '--'*15) 
	clf = lgb.train(
		params,
		dtrain,
		num_boost_round=20, 
		valid_sets=[dtrain, dtest]
		)

clf.params
print(clf.current_iteration(), clf.best_iteration, clf.num_trees(),sep='\n')

refitted = clf.refit(X_test,y_test) 
refitted.params
print(refitted.current_iteration(), refitted.best_iteration, refitted.num_trees(), sep='\n')
```

## Learning rate decay

学习率衰减，通过callback中添加reset_parameter传递学习率，接受两种参数类型：

1. num_boost_round 长度的 list
2. 以当前迭代次数为参数的函数 function(curr_iter)

```python
# reset_parameter callback accepts:
# 1. list with length = num_boost_round
# 2. function(curr_iter)
bst = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                valid_sets=lgb_eval,
                callbacks=[lgb.reset_parameter(learning_rate=lambda iter: 0.05 * (0.99 ** iter))])

# change other parameters during training
bst = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                valid_sets=lgb_eval,
                callbacks=[lgb.reset_parameter(bagging_fraction=[0.7] * 5 + [0.6] * 5)])
```



# LightGBM with PySpark

LightGBM 目前提供3种分布式学习算法：

| Parallel Algorithm | How to Use             |
| ------------------ | ---------------------- |
| Data parallel      | `tree_learner=data`    |
| Feature parallel   | `tree_learner=feature` |
| Voting parallel    | `tree_learner=voting`  |

这些算法适用于不同的场景：

|                   | #data is small   | #data is large  |
| ----------------- | ---------------- | --------------- |
| #feature is small | Feature Parallel | Data Parallel   |
| #feature is large | Feature Parallel | Voting Parallel |

Apache Spark users can use SynapseML for machine learning workflows with LightGBM. This project is not maintained by LightGBM’s maintainers.

See this SynapseML example for additional information on using LightGBM on Spark.

SynapseML原名MMLSpark

Simple and Distributed Machine Learning

SynapseML.svg

# Scikit-Learn API

| [`LGBMModel`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMModel.html#lightgbm.LGBMModel)([boosting_type, num_leaves, ...]) | Implementation of the scikit-learn API for LightGBM. |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [`LGBMClassifier`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier)([boosting_type, num_leaves, ...]) | LightGBM classifier.                                 |
| [`LGBMRegressor`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor)([boosting_type, num_leaves, ...]) | LightGBM regressor.                                  |
| [`LGBMRanker`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker)([boosting_type, num_leaves, ...]) | LightGBM ranker.                                     |

```python
import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Step 1: 
# load or create your dataset
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Step 2: Training
gbm = lgb.LGBMRegressor(num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        callbacks=[lgb.early_stopping(5)])


# Step 5:  Predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

# Step 6:  Evaluate
rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
print(f'The RMSE of prediction is: {rmse_test}')

# feature importances
print(f'Feature importances: {list(gbm.feature_importances_)}')
```



```python
# other scikit-learn modules
estimator = lgb.LGBMRegressor(num_leaves=31)

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}

gbm = GridSearchCV(estimator, param_grid, cv=3)
gbm.fit(X_train, y_train)

print(f'Best parameters found by grid search are: {gbm.best_params_}')
```



