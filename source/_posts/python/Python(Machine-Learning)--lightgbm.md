---
title: Python(Machine Learning)--LightGBM
tags:
  - Python
  - 机器学习
categories:
  - Python
  - Machine Learning
cover: /img/LightGBM_cover.svg
top_img: /img/LightGBM_cover.svg
abbrlink: '44910830'
katex: true
date: 2024-01-25 22:15:00
description:
---

# Quick Start

LightGBM（Light Gradient Boosting Machine）是一种高效的 Gradient Boosting 算法， 主要用于解决GBDT在海量数据中遇到的问题，以便更好更快的用于工业实践中。

在实际建模环节，LGBM支持Python、Java、C++等多种编程语言进行调用，并同时提供了Sklearn API和原生API两套调用方法。

使用原生LGBM API时需要先将数据集转化成一种LGBM库定义的一种特殊的数据格式 Dataset，然后以字典形式设置参数，最终使用LGBM中自带的方法`lgb.train`或`lgb.cv`进行训练。

| 数据结构                                                     | 说明                   |
| ------------------------------------------------------------ | ---------------------- |
| [`lightgbm.Dataset`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Dataset.html#lightgbm.Dataset) | LightGBM数据集         |
| [`lightgbm.Booster`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html#lightgbm.Booster) | LightGBM中的返回的模型 |
| `lightgbm.CVBooster`                                         | CVBooster in LightGBM  |

```python
lightgbm.Dataset(data, 
                 label=None, 
                 reference=None, 
                 weight=None, 
                 group=None, 
                 init_score=None, 
                 feature_name='auto', 
                 categorical_feature='auto', 
                 params=None, 
                 free_raw_data=True)
```

常用参数：

- data 内部数据集的数据源
- label 数据标签
- reference 在lightgbm中验证数据集应使用训练数据集作为参考。
- weight 每个样本的权重
- feature_name (list of str, or 'auto') 特征名称，默认 auto，如果数据是pandas.DataFrame，则使用数据列名称。
- categorical_feature (list of str, or 'auto') 分类特征名称。
- free_raw_data 如果为True，则在构建内部数据集后释放原始数据。如果想重复使用 Dataset ，则设为 False

```python
import lightgbm as lgb
```

现在，我们来简单看看原生代码是如何实现的。

[Jupyter Notebook Demo](/ipynb/classification_demo.html#LightGBM)

## Step 1:  Load the dataset

```python
# load or create your dataset
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X, y = load_boston(as_frame=True, return_X_y=True)
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
dtrain = lgb.Dataset(X_train, y_train, feature_name=['c1', 'c2', 'c3'], categorical_feature=['c3'])
```

## Step 2: Setting Parameters

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

## Step 3: Training

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

## Step 4: Save and load model

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

## Step 5:  Predict

```python
# A model that has been trained or loaded can perform predictions on datasets:
y_pred = bst.predict(X_test)

# If early stopping is enabled during training, you can get predictions from the best iteration with bst.best_iteration:
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
```

## Step 6:  Evaluating

```python
from sklearn.metric import mean_squared_error
rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
print(f'The RMSE of prediction is: {rmse_test}')
```

# 参数

lightgbm 的参数以 dict 的格式配置，然后训练的时候传递给 lightgbm.train 的 params 参数。接下来我们就逐个解释这些参数，并对其使用方法进行说明。

## 基本参数

**task**：指定任务类型。default = `train`，aliases：`task_type`

- `train` 用于训练，alias：`training`
- `predict` 用于预测，alias：`prediction`，`test`
- `convert_model` 将模型文件转换为if-else格式
- `refit` 用新数据刷新现有模型，alias：`refit_tree`
- `save_binary` 将数据集保存到二进制文件中

**boosting**：指定算法类型。default = `gbdt`, aliases: `boosting_type`, `boost`

- `gbdt`：传统的梯度提升算法，是最常用、且性能最稳定的 boosting 类型。alias：`gbrt`。
- `rf`：传统的梯度促进决策树，alias：`random_forest`
- `dart`： (Dropouts meet Multiple Additive Regression Trees)是一种结合了 Dropout 和多重加性回归树的方法。它在每次迭代过程中随机选择一部分树进行更新，会较大程度增加模型随机性，可以用于存在较多噪声的数据集或者数据集相对简单（需要减少过拟合风险）的场景中

**objective**：指定目标函数。str or callable, default = regression

- 回归问题：`regression`, `regression_l1`, `huber`, `fair`, `poisson`, `quantile`, `mape`, `gamma`, `tweedie`
- 分类问题：`binary`, `multiclass`,  `multiclassova` 。对于多分类`num_class`参数也应该设置
- 交叉熵：`cross_entropy`, `cross_entropy_lambda`, 
- 排序问题：`lambdarank`, `rank_xendcg`

**num_class**：仅用于 `multi-class` default = 1

**data_sample_strategy**：default = bagging

- `bagging`：机装袋取样。注意，当 bagging_freq > 0 且 bagging_fraction < 1.0 时起作用。
- `goss`：（Gradient-based One-Side Sampling）是一种基于梯度的单侧采样方法。它在每次迭代中只使用具有较大梯度的样本进行训练，适用于大规模数据集，可以在保持较高精度的同时加速训练过程。

**num_threads**：并行的线程数。default = 0, aliases: `num_thread`, `nthread`, `nthreads`, `n_jobs`

**device_type**：学习设备。default = `cpu`, options: `cpu`, `gpu`, `cuda`, aliases: `device`

**seed**：随机种子。default = `None`, aliases: `random_seed`, `random_state`

**verbosity**：日志输出详细程度，default = 1 。aliases: `verbose`

- `< 0` 仅输出致命错误
- `= 0`显示警告和报错
- `= 1` 用于打印全部信息
- `> 1` Debug

## 样本处理参数

| Name                | Description                                                  | aliases                    |
| :------------------ | :----------------------------------------------------------- | -------------------------- |
| is_unbalance        | 是否不平衡数据集，仅用于binary和multiclassova。默认 False    | unbalance, unbalanced_sets |
| scale_pos_weight    | 调整正样本权重，仅用于分类任务。默认1.0                      |                            |
| categorical_feature | 识别分类特征名称。e.g. `categorical_feature=0,1,2 `  or `categorical_feature=name:c1,c2,c3` |                            |

## 特征处理参数

| Name                          | Description                                                  | aliases           |
| :---------------------------- | :----------------------------------------------------------- | ----------------- |
| bin_construct_sample_cnt      | 该参数表示对连续变量进行分箱时（直方图优化过程）抽取样本的个数，默认取值为200000 | subsample_for_bin |
| saved_feature_importance_type | 特征重要性计算方式，默认为 0，表示在模型中被选中作为分裂特征的次数，可选1，表示在模型中的分裂增益之和作为重要性评估指标 |                   |
| max_cat_threshold             | 分类特征的最大拆分点数量，默认值为32                         |                   |
| cat_l2                        | 分类特征L2 正则化系数，默认值为10.0                          |                   |
| cat_smooth                    | 减少分类特征中噪声的影响，特别是对于数据很少的类别，默认值为10.0 |                   |
| max_cat_to_onehot             | 当分类特征类别数小于或等于max_cat_to_onehot 时将使用其他拆分算法 |                   |

## 决策树生成

| Name                    | Description                                                  | aliases                                                      |
| :---------------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| max_depth               | 树的最大深度，默认值为 -1，表示无限制                        |                                                              |
| num_leaves              | 一棵树上的叶子节点数，默认值为 31                            | num_leaf, max_leaves, max_leaf, max_leaf_nodes               |
| min_data_in_leaf        | 单个叶子节点上的最小样本数量，默认值为 20。较大的值可以防止过拟合。 | min_data_per_leaf, min_data, min_child_samples, min_samples_leaf |
| min_sum_hessian_in_leaf | 一片叶子节点的最小权重和，默认值为 1e-3。较大的值可以防止过拟合。 | min_sum_hessian_per_leaf, min_sum_hessian, min_hessian, min_child_weight |
| bagging_fraction        | 训练时的抽样比例，默认值为 1.0。对于二分类问题，还可控制正负样本抽样比例 pos_bagging_fraction 和 neg_bagging_fraction | sub_row, subsample, bagging                                  |
| bagging_freq            | 抽样频率，表示每隔几轮进行一次样本抽样，默认取值为0，表示不进行随机抽样。 | subsample_freq                                               |
| feature_fraction        | 在每次迭代（树的构建）时，随机选择特征的比例，取值范围为 (0, 1]，默认为1.0。 | sub_feature, colsample_bytree                                |
| feature_fraction_bynode | 每个树节点上随机选择一个特征子集，默认为1.0。                | sub_feature_bynode, colsample_bynode                         |
| extra_trees             | 极端随机树。默认为 False，如果设置为True，在节点拆分时，LightGBM将只为每个特征选择一个随机选择的阈值 |                                                              |
| min_gain_to_split       | 再分裂所需最小增益，默认值为 0，表示无限制                   | min_split_gain                                               |

注意：feature_fraction 不受subsample_freq影响。同时需要注意的是，LGBM和随机森林不同，随机森林是每棵树的每次分裂时都随机分配特征，而LGBM是每次构建一颗树时随机分配一个特征子集，这颗树在成长过程中每次分裂都是依据这个特征子集进行生长。

## 模型训练

| Name               | Description                                                  | aliases                                                      |
| :----------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| data               | 用于训练的数据集                                             | train, train_data, train_data_file, data_filename            |
| valid              | 验证/测试数据，支持多个验证数据，使用逗号`,`分隔             | test, valid_data, valid_data_file, test_data, test_data_file, valid_filenames |
| num_iterations     | 提升迭代次数，即生成的基学习器的数量，默认值100。注意：对于多分类问题，树的数量等于 num_class * num_iterations | num_iteration, n_iter, num_tree, num_trees, num_round, num_rounds, nrounds, num_boost_round, n_estimators, max_iter |
| learning_rate      | 学习率，即每次迭代中梯度提升的步长，默认值0.1。              | shrinkage_rate, eta                                          |
| lambda_l1          | L1 正则化系数，默认值为 0                                    | reg_alpha, l1_regularization                                 |
| lambda_l2          | L2 正则化系数，默认值为 0                                    | reg_lambda, lambda, l2_regularization                        |
| metric             | 评估指标，默认“”                                             | metrics, metric_types                                        |
| min_data_per_group | 每个分类组的最小数据数量，默认值为 100                       |                                                              |
| input_model        | 对于prediction任务，该模型将用于预测；对于train任务，将从在这个模型基础上继续训练 | model_input, model_in                                        |

损失函数

$$
Obj_k = \sum_{i=1}^Nl(y_i,\hat{y_i}) + \gamma T + \frac{1}{2}\lambda\sum_{j=1}^Tw_j^2 + \alpha\sum_{j=1}^Tw_j
$$

其中$T$表示当前第$k$棵树上的叶子总量，$w_j$则代表当前树上第$j$片叶子的叶子权重（leaf weights），即当前叶子$j$的预测值。正则项有两个：使用平方的 $\ell_2$正则项与使用绝对值的 $\ell_1$正则项。

部分参数在可模型训练 lightgbm.train 时传递值：

```python
lightgbm.train(params, 
               train_set, 
               num_boost_round=100, 
               valid_sets=None, 
               valid_names=None, 
               feval=None, 
               init_model=None, 
               feature_name='auto', 
               categorical_feature='auto', 
               keep_training_booster=False, 
               callbacks=None)

lightgbm.cv(params, 
            train_set, 
            num_boost_round=100, 
            folds=None, nfold=5, 
            stratified=True, 
            shuffle=True, 
            metrics=None, 
            feval=None, 
            init_model=None, 
            feature_name='auto', 
            categorical_feature='auto', 
            fpreproc=None, 
            seed=0, 
            callbacks=None, 
            eval_train_metric=False, 
            return_cvbooster=False)
```

注意：通过 params (dict) 传递的值优先于通过参数提供的值。

其中 feval 用来自定义评估函数。

```python
# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: str, eval_result: float, is_higher_better: bool
# Relative Absolute Error (RAE)
def rae(preds, train_data):
    labels = train_data.get_label()
    return 'RAE', np.sum(np.abs(preds - labels)) / np.sum(np.abs(np.mean(labels) - labels)), False

# Starting training with custom eval functions...
lgb.train(dtrain
        valid_sets=[dtrain, dtest],
        feval=rae,
        callbacks=[lgb.early_stopping(5)])
```

> 注意：sklearn API 自定义评估函数有所不同 `f(y_true, y_pred) -> name, eval_result, is_higher_better` 。调用 `fit` 方法时传递给 eval_metric 参数。

## 回调参数

callbacks 参数标识在每次迭代中应用的回调函数列表。

| 方法                                           | Create a callback                                            |
| ---------------------------------------------- | ------------------------------------------------------------ |
| `lightgbm.early_stopping(stopping_rounds)`     | 回调提前停止策略，控制过拟合风险，当验证集上的精度若干轮不下降，提前停止训练。 |
| `lightgbm.log_evaluation([period, show_stdv])` | 输出评估结果的频率                                           |
| `lightgbm.record_evaluation(eval_result)`      | 在`eval_result`中记录评估结果                                |
| `lightgbm.reset_parameter(**kwargs)`           | 第一次迭代后重置参数                                         |

lightgbm 可通过在callback中添加reset_parameter传递学习率，从而实现学习率衰减(learning rate decay)。

学习率接受两种参数类型：

1. num_boost_round 长度的 list
2. 以当前迭代次数为参数的函数 function(curr_iter)

```python
# reset_parameter callback accepts:
# 1. list with length = num_boost_round
# 2. function(curr_iter)
bst = lgb.train(params,
                dtrain,
                num_boost_round=10,
                init_model=gbm,
                valid_sets=deval,
                callbacks=[lgb.reset_parameter(learning_rate=lambda iter: 0.05 * (0.99 ** iter))])

# change other parameters during training
bst = lgb.train(params,
                dtrain,
                num_boost_round=10,
                init_model=gbm,
                valid_sets=deval,
                callbacks=[lgb.reset_parameter(bagging_fraction=[0.7] * 5 + [0.6] * 5)])
```

# Scikit-Learn API

LGBM的 sklearn API支持使用sklearn的调用风格和语言习惯进行LGBM模型训练，数据读取环节支持直接读取本地的Numpy或Pandas格式数据，而在实际训练过程中需要先实例化评估器并设置超参数，然后通过.fit的方式进行训练，并且可以直接调用grid search进行超参数搜索，也可以使用其他sklearn提供的高阶工具，如构建机器学习流、进行特征筛选或者进行模型融合等。

总的来看，LGBM的sklearn API更加轻量、便捷，并且能够无缝衔接sklearn中其他评估器，快速实现sklearn提供的高阶功能，对于熟悉sklearn的用户而言非常友好；而原生API则会复杂很多，但同时也提供了大量sklearn API无法实现的复杂功能，若能够合理使用，则可以实现相比sklearn API更精准的建模结果、更高效的建模流程。

| module                                                       | comment                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [`LGBMModel`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMModel.html#lightgbm.LGBMModel) | Implementation of the scikit-learn API for LightGBM. |
| [`LGBMClassifier`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier) | LightGBM classifier.                                 |
| [`LGBMRegressor`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor) | LightGBM regressor.                                  |
| [`LGBMRanker`](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html#lightgbm.LGBMRanker) | LightGBM ranker.                                     |

其中LGBMModel是 LightGBM 的基本模型类，它是一个泛型模型类，可以用于各种类型的问题（如分类、回归等）。通常，我们不直接使用 LGBMModel，而是使用针对特定任务的子类使用不同的类，即分类问题使用 LGBMClassifier 、回归问题使用 LGBMRegressor，而排序问题则使用LGBMRanker。

以 LGBMClassifier 为例，默认参数如下：

```python
LGBMClassifier(
    boosting_type: str = 'gbdt',
    num_leaves: int = 31,
    max_depth: int = -1,
    learning_rate: float = 0.1,
    n_estimators: int = 100,
    subsample_for_bin: int = 200000,
    objective: Union[str, Callable, NoneType] = None,
    class_weight: Union[Dict, str, NoneType] = None,
    min_split_gain: float = 0.0,
    min_child_weight: float = 0.001,
    min_child_samples: int = 20,
    subsample: float = 1.0,
    subsample_freq: int = 0,
    colsample_bytree: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    random_state: Union[int, numpy.random.mtrand.RandomState, NoneType] = None,
    n_jobs: int = -1,
    silent: Union[bool, str] = 'warn',
    importance_type: str = 'split',
    **kwargs,
)
```

具体的模型训练过程和sklearn中其他模型一样，通过fit进行训练，并利用predict进行结果输出：

```python
import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Step 1: load or create your dataset
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
y_score = gbm.predict_proba(X_test num_iteration=gbm.best_iteration_)

# Step 6:  Evaluate
rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
print(f'The RMSE of prediction is: {rmse_test}')

# feature importances
print(f'Feature importances: {list(gbm.feature_importances_)}')
```

可以与sklearn中其他方法无缝衔接：

```python
# other scikit-learn modules
from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}

gbm = GridSearchCV(estimator, param_grid, cv=3)
gbm.fit(X_train, y_train)

print(f'Best parameters found by grid search are: {gbm.best_params_}')
```

# 自定义损失函数

## 原生接口

在lgb.train中通过参数 `params["objective"]` 和 `feval` 来自定义损失函数和评估函数。

> 老版本lightgbm自定义损失函数需要通过lgb.train中的参数fobj传递，最新版本改为直接在配置params时通过objective传递，fobj参数已经废弃。

[advanced_example.py](https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py)

注意：

1. 在LightGBM中，自定义损失函数需要返回损失函数的一阶(grad)和二阶(hess)导数。
2. 自定义损失函数后，模型的输出不在是 [0,1] 概率输出，而是 sigmoid 函数之前的输入值。
3. 自定义损失函数后，模型的输出已经发生改变，需要写出对应的评估函数。
4. 自定义损失函数后，LightGBM默认的boost_from_average=True失效，按照GBDT的框架，对于利用logloss来优化的二分类问题，样本的初始值为训练集标签的均值，在自定义损失函数后,系统无法获取到这个初始化值，导致收敛速度变慢。可以在构建lgb.Dataset时，利用init_score参数手动完成。
5. 自定义损失函数后，模型输出需要手动进行sigmoid函数变换

- 损失函数： `f(preds, train_data) -> grad, hess` ，配置在lgb.train的params字典中，使用 objective 参数传递。
- 评估函数： `f(preds, train_data) -> name, eval_result, is_higher_better` ，使用lgb.train的 feval 参数传递。

```python
# NOTE: when you do customized loss function, the default prediction value is margin
# This may make built-in evaluation metric calculate wrong results
# For example, we are doing log likelihood loss, the prediction is score before logistic transformation
# Keep this in mind when you use the customization

# self-defined objective function
# f(preds: array, train_data: Dataset) -> grad: array, hess: array
# log likelihood loss
from scipy import special

def loglikelihood(preds, train_data):
    labels = train_data.get_label()
    preds = special.expit(preds)
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess

# Pass custom objective function through params
params = {"objective": loglikelihood}

# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: str, eval_result: float, is_higher_better: bool
def binary_error(preds, train_data):
    labels = train_data.get_label()
    preds = special.expit(preds)
    return "error", np.mean(labels != (preds > 0.5)), False

gbm = lgb.train(
    params, train_data, num_boost_round=100, feval=binary_error, valid_sets=test_data
)

y_pred = special.expit(gbm.predict(X_test))
```

## sklearn API

sklearn API 自定义损失函数和评估函数和原生接口有所不同。

- 损失函数： `f(y_true, y_pred) -> grad, hess` ，新建sklearn模型实例时使用 objective 参数传递。
- 评估函数： `f(y_true, y_pred) -> name, eval_result, is_higher_better` ，调用 `fit` 方法时传递给 eval_metric 参数。

```python
# self-defined objective function

# f(y_true: array, y_pred: array) -> grad: array, hess: array
# log likelihood loss
from scipy import special

def loglikelihood(y_true, y_pred):
    y_pred = special.expit(y_pred)
    grad = y_pred - y_true
    hess = y_pred * (1.0 - y_pred)
    return grad, hess

# Pass custom objective function through objective
model = LGBMModel(objective=loglikelihood, n_estimators=100)

# self-defined eval metric
# f(y_true: array, y_pred: array) -> name: str, eval_result: float, is_higher_better: bool
def binary_error(y_true, y_pred):
    y_pred = special.expit(y_pred)
    return "error", np.mean(y_true != (y_pred > 0.5)), False

model.fit(
    X_train, y_train, eval_metric=binary_error, eval_set=[(X_test, y_test)]
)

y_pred = special.expit(model.predict_proba(X_test))
```


# 可视化

| module                                       | comment                        |
| -------------------------------------------- | ------------------------------ |
| plot_importance(booster)                     | 绘制模型的特征重要性。         |
| plot_split_value_histogram(booster, feature) | 绘制模型指定特征的拆分值直方图 |
| plot_metric(booster)                         | 绘制训练期间的模型得分         |
| plot_tree(booster)                           | 绘制指定的树                   |
| create_tree_digraph(booster)                 | 创建指定树的二叉图文件         |

```python
evals_result = {}  # to record eval results for plotting
gbm = lgb.train(
    params,
    dtrain,
    num_boost_round=100,
    valid_sets=[dtrain, deval],
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

# 继续训练

lightGBM有两种增量学习方式：

[Jupyter notebook 增量学习Demo](/ipynb/incremental_learning_demo.html#LightGBM)

1. **init_model参数**：如果 init_model不为None，将从这个模型基础上继续训练，添加 num_boost_round 棵新树

```python
# init_model accepts:
# 1. model file name
# 2. Booster()

bst = lgb.train(
    previous_params,
    new_data,
    num_boost_round=10,
    init_model=previous_model, 
    valid_sets=eval_data,
    keep_training_booster=True
)
```

其中 keep_training_booster (bool) 参数表示返回的模型 (booster) 是否将用于保持训练，默认False。当模型非常大并导致内存错误时，可以尝试将此参数设置为True，以避免 model_to_string 转换。然后仍然可以使用返回的booster作为init_model，用于未来的继续训练。

2. **调用 refit 方法**：在原有模型的树结构都不变的基础上，重新拟合新数据更新叶子节点权重

在参数字典中配置

```python
params = {
	'task':'refit', 
	'refit_decay_rate': 0.9,
	'boosting_type':'gbdt',
	'objective':'binary',
	'metric':'auc'
}

bst = lgb.train(
		params,
		dtrain,
		num_boost_round=20, 
		valid_sets=[dtrain, deval],
    init_model=previous_model 
)
```

或用返回的模型 (Booster) 重新拟合

```python
bst.refit(
    data=X_train,
    label=y_train,
    decay_rate=0.9
  )
```

其中 refit_decay_rate 控制 refit 任务中叶节点的衰减率。重新拟合后，叶子结点的输出的计算公式为

```
leaf_output = refit_decay_rate * old_leaf_output + (1.0 - refit_decay_rate) * new_leaf_output
```

# 分布式学习

LGBM还提供了分布式计算版本和GPU计算版本进行加速计算，其中分布式计算模式下支持从HDFS（Hadoop Distributed File System）系统中进行数据读取和计算，而GPU计算模式下则提供了GPU version（借助OpenCL，即Open Computing Language来实现多种不同GPU的加速计算）和CUDA version（借助CUDA，即Compute Unified Device Architecture来实现NVIDIA GPU加速）。不过，不同于深度学习更倾向于使用CUDA加速，对于LGBM而言，由于目前CUDA version只能在Linux操作系统下实现，因此大多数情况下，我们往往会选择支持Windows系统的GPU version进行GPU加速计算。

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

**tree_learner** 参数控制分布式学习方法。default = serial,  aliases: `tree`, `tree_type`, `tree_learner_type`

- serial：单机学习
- feature：特征并行，别名：feature_parallel
- data：数据并行，别名：data_parallel
- voting：投票平行，别名：voting_parallel

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/python/SynapseML.svg" height="80%;" align="right"/> 要在spark上使用LightGBM，需要安装[SynapseML](https://microsoft.github.io/SynapseML/)包，原名MMLSpark，由微软开发维护。SynapseML建立在Apache Spark分布式计算框架上，与SparkML/MLLib库共享相同的API，允许您将SynapseML模型无缝嵌入到现有的Apache Spark工作流程中。

SynapseML在Python中安装：首先，默认已经安装好了PySpark，然后，通过pyspark.sql.SparkSession配置会自动下载并安装到现有的Spark集群上

```python
import pyspark
# Use 0.11.4-spark3.3 version for Spark3.3 and 1.0.2 version for Spark3.4
spark = pyspark.sql.SparkSession.builder.appName("MyApp") \
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.0.2") \
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
            .getOrCreate()
import synapse.ml
```

或者通过启动Spark时配置`--packages`选项

```python
# Use 0.11.4-spark3.3 version for Spark3.3 and 1.0.2 version for Spark3.4
spark-shell --packages com.microsoft.azure:synapseml_2.12:1.0.2
pyspark --packages com.microsoft.azure:synapseml_2.12:1.0.2
spark-submit --packages com.microsoft.azure:synapseml_2.12:1.0.2 MyApp.jar
```

这个包比较大，第一次安装需要较长时间。

| 算法               | 说明                                                         |
| ------------------ | ------------------------------------------------------------ |
| LightGBMClassifier | 用于构建分类模型。例如，为了预测公司是否破产，我们可以使用LightGBMClassifier构建一个二进制分类模型。 |
| LightGBMRegressor  | 用于构建回归模型。例如，为了预测房价，我们可以用LightGBMRegressor建立一个回归模型。 |
| LightGBMRanker     | 用于构建排名模型。例如，为了预测网站搜索结果的相关性，我们可以使用LightGBMRanker构建一个排名模型。 |

在PySpark中，您可以通过以下方式运行`LightGBMClassifier`：

```python
from synapse.ml.lightgbm import LightGBMClassifier
model = LightGBMClassifier(
    learningRate=0.3,
    numIterations=100,
    numLeaves=31
).fit(train)
```

LightGBM的参数比SynapseML公开的要多得多，若要添加额外的参数，请使用passThroughArgs字符串参数配置。

```python
from synapse.ml.lightgbm import LightGBMClassifier
model = LightGBMClassifier(
    passThroughArgs="force_row_wise=true min_sum_hessian_in_leaf=2e-3",
    numIterations=100,
    numLeaves=31
).fit(train)
```

您可以混合passThroughArgs和显式args，如示例所示。SynapseML合并它们以创建一个要发送到LightGBM的参数字符串。如果您在两个地方都设置参数，则以passThroughArgs为优先。

[Jupyter notebook 分布式学习Demo](/ipynb/distributed_learning_demo.html#LightGBM-with-spark)

示例：

```python
# Read dataset
from synapse.ml.core.platform import *
df = (
    spark.read.format("csv")
    .option("header", True)
    .option("inferSchema", True)
    .load(
        "wasbs://publicwasb@mmlspark.blob.core.windows.net/company_bankruptcy_prediction_data.csv"
    )
)
# print dataset size
print("records read: " + str(df.count()))
print("Schema: ")
df.printSchema()
display(df)

# Split the dataset into train and test
train, test = df.randomSplit([0.85, 0.15], seed=1)

# Add featurizer to convert features to vector
from pyspark.ml.feature import VectorAssembler

feature_cols = df.columns[1:]
featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_data = featurizer.transform(train)["Bankrupt?", "features"]
test_data = featurizer.transform(test)["Bankrupt?", "features"]

# Check if the data is unbalanced
display(train_data.groupBy("Bankrupt?").count())

# Model Training
from synapse.ml.lightgbm import LightGBMClassifier

model = LightGBMClassifier(
    objective="binary", featuresCol="features", labelCol="Bankrupt?", isUnbalance=True
)
model = model.fit(train_data)

# "saveNativeModel" allows you to extract the underlying lightGBM model for fast deployment after you train on Spark.
from synapse.ml.lightgbm import LightGBMClassificationModel

if running_on_synapse():
    model.saveNativeModel("/models/lgbmclassifier.model")
    model = LightGBMClassificationModel.loadNativeModelFromFile(
        "/models/lgbmclassifier.model"
    )
if running_on_synapse_internal():
    model.saveNativeModel("Files/models/lgbmclassifier.model")
    model = LightGBMClassificationModel.loadNativeModelFromFile(
        "Files/models/lgbmclassifier.model"
    )
else:
    model.saveNativeModel("/tmp/lgbmclassifier.model")
    model = LightGBMClassificationModel.loadNativeModelFromFile(
        "/tmp/lgbmclassifier.model"
    )

# Feature Importances Visualization
import pandas as pd
import matplotlib.pyplot as plt

feature_importances = model.getFeatureImportances()
fi = pd.Series(feature_importances, index=feature_cols)
fi = fi.sort_values(ascending=True)
f_index = fi.index
f_values = fi.values

# print feature importances
print("f_index:", f_index)
print("f_values:", f_values)

# plot
x_index = list(range(len(fi)))
x_index = [x / len(fi) for x in x_index]
plt.rcParams["figure.figsize"] = (20, 20)
plt.barh(
    x_index, f_values, height=0.028, align="center", color="tan", tick_label=f_index
)
plt.xlabel("importances")
plt.ylabel("features")
plt.show()

# Model Prediction
predictions = model.transform(test_data)
predictions.limit(10).toPandas()

from synapse.ml.train import ComputeModelStatistics

metrics = ComputeModelStatistics(
    evaluationMetric="classification",
    labelCol="Bankrupt?",
    scoredLabelsCol="prediction",
).transform(predictions)
display(metrics)
```

# 分布式预测

当我们训练好一个本地模型，想在大规模的数据上预测时，可以使用pandas_udf进行分布式预测：

```python
from pyspark.sql.functions import pandas_udf, struct
import joblib 
import pandas as pd

def predict_with_spark(spark_df, spark_context, local_model):
   var = spark_context.broadcast(local_model)
   model = var.value

   @pandas_udf('float')
   def transform(X):
      return pd.Series(model.predict_proba(X)[:, 1])

   cols = struct(*model.feature_name())
   return spark_df.withColumn('predictions', transform(cols))

clf = joblib.load('clf.pkl')
df = spark.sql("select * from home_credit_default_risk")
predict_with_spark(df, sc, clf).select('predictions').show()
```

