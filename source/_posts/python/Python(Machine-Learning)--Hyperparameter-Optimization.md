---
title: Python(Machine Learning)--超参数优化
tags:
  - Python
  - 机器学习
categories:
  - Python
  - 'Machine Learning'
cover: /img/Hyperparameter-Optimization.png
top_img: /img/optuna-logo.png
description: 
abbrlink: 794d8498
date: 2024-02-15 11:28:52
---

# 超参数优化

超参数是用于控制学习过程的不同参数值，对机器学习模型的性能有显著影响。例如，随机森林算法中的估计器数量、最大深度和分裂标准等。超参数优化是找到超参数值的正确组合，以便在合理的时间内实现数据最大性能的过程。这个过程在机器学习算法的预测准确性中起着至关重要的作用。因此，超参数优化被认为是构建机器学习模型中最棘手的部分。

目前来说sklearn支持两种类型的超参数优化：

- GridSearchCV 网格搜索是一种广泛使用的传统方法，详尽地考虑了所有参数组合
- RandomizedSearchCV 随机搜索可以从具有指定分布的参数空间中抽样给定数量的候选者

贝叶斯优化方法 (Bayesian Optimization)是当前超参数优化领域的SOTA手段（State of the Art），可以被认为是当前最为先进的优化框架。

贝叶斯优化的工作原理是：首先对目标函数的全局行为建立先验知识（通常用高斯过程来表示），然后通过观察目标函数在不同输入点的输出，更新这个先验知识，形成后验分布。基于后验分布，选择下一个采样点，这个选择既要考虑到之前观察到的最优值（即利用），又要考虑到全局尚未探索的区域（即探索）。这个选择的策略通常由所谓的采集函数（Acquisition Function）来定义，比如最常用的期望提升（Expected Improvement），这样，贝叶斯优化不仅可以有效地搜索超参数空间，还能根据已有的知识来引导搜索，避免了大量的无用尝试。

具体的算法细节可以参考：https://zhuanlan.zhihu.com/p/643095927?utm_id=0

本文介绍一些实用的超参数优化技术：

1. Hyperopt
2. Scikit Optimize
3. Optuna

```python
# read the dataset
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

# Hyperopt

Hyperopt优化器是目前最通用的贝叶斯优化器之一，它集成了包括随机搜索、模拟退火和TPE（Tree-structured Parzen Estimator Approach）等多种优化算法。

官方文档：https://hyperopt.github.io/hyperopt/

安装库

```sh
pip install hyperopt
```

Hyperopt 优化过程主要分为4步：

Step 1 定义参数空间
Step 2 定义目标函数
Step 3 执行优化
Step 4 评估输出

## Step 1 定义参数空间

我们使用`dict()`来定义超参数空间，其中key可以任意设置，value则需用hyperopt的hp函数：

| hyperopt.hp                         | 说明                                     |
|:----------------------------------- |:-------------------------------------- |
| hp.choice(label, options)           | 用于分类参数，返回options 中的元素                  |
| hp.pchoice(label, p_list)           | 返回 (probability, option) 元素对           |
| hp.randint(label, low, high)        | 返回区间 [low, upper) 内的随机整数               |
| hp.uniform(label, low, high)        | 均匀返回 low, high 之间的浮点数                  |
| hp.quniform(label, low, high, q)    | 均匀返回 low, high 之间的浮点数，适用于离散值           |
| hp.uniformint(label, low, high)     | 均匀返回 low, high 之间均的整数，适用于离散值           |
| hp.loguniform(label, low, high)     | 对数均匀返回 e^low^,e^high^ 之间浮点数            |
| hp.qloguniform(label, low, high, q) | 对数均匀返回    e^low^, e^high^ 之间浮点数，适用于离散值 |
| hp.normal(label, mu, sigma)         | 正态分布返回实数                               |
| hp.qnormal(label, mu, sigma, q)     | 正态分布返回实数，适用于离散值                        |
| hp.lognormal(label, mu, sigma)      | 对数正态分布返回实数                             |
| hp.qlognormal(label, mu, sigma, q)  | 正态分布返回实数，适用于离散值                        |

> 每个hp函数都有一个label作为第一个参数，这些label用于在优化过程中将参数传递给调用方。

```python
# define a search space
from hyperopt import hp
space = {
    'random_state': 42, 
    'max_depth': hp.uniformint('max_depth', 2, 10),
    'learning_rate': hp.loguniform('learning_rate', 0.001, 1.0),
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400]),
    'subsample': hp.quniform('subsample', 0.1, 1.0, 0.1),
    'max_features': hp.choice('max_features', ['sqrt', 'log2'])
}
```

## Step 2 定义目标函数

Hyperopt 目前只支持目标函数的最小化

```python
# define an objective function
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

def objective(params):    
    reg = GradientBoostingRegressor(**params)
    mse = cross_val_score(reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5).mean()
    return -1*mse
```

## Step 3 执行优化

hyperopt 使用 fmin 函数进行优化。

fmin接收两种搜索算法：

- tpe.suggest 指代TPE (Tree Parzen Estimators) 方法
- rand.suggest 指代随机网格搜索方法

```python
# minimize the objective over the space
from hyperopt import fmin, tpe, space_eval, Trials 
trials = Trials() # Initialize trials object

best = fmin(
    fn=objective, # Objective Function to optimize
    space=space, # Hyperparameter's Search Space
    algo=tpe.suggest, # Optimization algorithm
    max_evals=100, # Number of optimization attempts
    trials = trials,
      verbose=True,
      early_stop_fn=no_progress_loss(5)
)
```

Output: 

```
100%|██████| 1000/1000 [02:35<00:00,  6.44trial/s, best loss: 8.932729710763638]
```

其中 Trials 对象用于保存所有的超参数、损失和其他信息。

## Step 4 评估输出

```python
print(space_eval(space, best))
```

Output: 

```
{'learning_rate': 0.2, 'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 54, 'subsample': 0.9}
```

## 分布式优化

[并行化 Hyperopt 超参数优化](https://learn.microsoft.com/zh-cn/azure/databricks/machine-learning/automl-hyperparam-tuning/hyperopt-spark-mlflow-integration?source=recommendations)

超参数调优通常涉及训练数百或数千个模型，Hyperopt 允许分布式调优。通过 trials 参数将 SparkTrials 传递给 fmin 函数，在Spark集群上并行运行这些任务。

```python
# We can run Hyperopt locally (only on the driver machine)
# by calling `fmin` without an explicit `trials` argument.
best_hyperparameters = fmin(
  fn=train,
  space=search_space,
  algo=algo,
  max_evals=32)

# We can distribute tuning across our Spark cluster
# by calling `fmin` with a `SparkTrials` instance.
from hyperopt import SparkTrials
spark_trials = SparkTrials()
best_hyperparameters = fmin(
  fn=train,
  space=search_space,
  algo=algo,
  trials=spark_trials,
  max_evals=32)
```

SparkTrials可以通过3个参数进行配置，所有这些参数都是可选的：

- parallelism 最大并行数，默认为 SparkContext.defaultParallelism。
- timeout 允许的最大时间（以秒为单位），默认为None。
- spark_session 如果没有给出，SparkTrials将寻找现有的SparkSession。

除了单机训练算法（例如 scikit-learn 中的算法）以外，还可以将 Hyperopt 与分布式训练算法配合使用。将 Hyperopt 与分布式训练算法配合使用时，请不要将 `trials` 参数传递给 `fmin()`，尤其是不要使用 `SparkTrials` 类。 `SparkTrials` 旨在为本身不是分布式算法的算法分配试运行。 对于分布式训练算法，请使用在群集驱动程序上运行的默认 `Trials` 类。 Hyperopt 评估驱动程序节点上的每个试运行，使 ML 算法本身可以启动分布式训练。

[将分布式训练算法与 Hyperopt 配合使用](https://learn.microsoft.com/zh-cn/azure/databricks/machine-learning/automl-hyperparam-tuning/hyperopt-distributed-ml?source=recommendations)

# Scikit-optimize

Scikit-optimize 建立在 Scipy、Numpy 和 Scikit-Learn之上。非常易于使用，它提供了用于贝叶斯优化的通用工具包，可用于超参数调优。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/python/image-20240220234034290.png" style="zoom:10%;" />

官方文档：https://scikit-optimize.github.io/stable/

安装库

```sh
pip install scikit-optimize
```

Scikit-optimize 优化过程主要分为4步：

Step 1 定义参数空间
Step 2 定义目标函数
Step 3 执行优化
Step 4 评估输出

## Step 1 定义参数空间

使用 Scikit-optimize 提供的方法定义参数空间：

| skopt.space                                | comment |
|:------------------------------------------ |:------- |
| space.Real(low, high,  prior, name)        | 用于浮点数参数 |
| space.Integer(low, high,  prior, name)     | 用于整数参数  |
| space.Categorical(categories, prior, name) | 用于分类参数  |

通过可选的prior参数可以对整型或浮点型取对数操作，或给类别型先验概率

```python
# define a search space
import skopt 
search_space= [
    skopt.space.Integer(2, 10, name='max_depth'),
    skopt.space.Real(0.001, 1.0, prior='log-uniform', name='learning_rate'),
    skopt.space.Integer(10, 100, name='n_estimators'),
    skopt.space.Real(0.2, 0.9, name='subsample'),
    skopt.space.Categorical(['sqrt', 'log2'], name='max_features')
]
```

## Step 2 定义目标函数

Scikit-optimize 支持目标函数最小化。

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from skopt.utils import use_named_args

# define the function used to evaluate a given configuration
@use_named_args(space)
def objective(**params):
    # configure the model with specific hyperparameters
    clf = GradientBoostingRegressor( random_state=42, **params)
    mse = cross_val_score(reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5).mean()
    return -1*mse
```

一般使用交叉验证来避免过拟合。used_named_args装饰器允许目标函数将参数作为关键字参数接收。

## Step 3 执行优化

有四种优化算法可供选择：

| skopt.optimizer | 说明           |
|:--------------- |:------------ |
| dummy_minimize  | 随机搜索         |
| forest_minimize | 使用决策树的贝叶斯优化  |
| gbrt_minimize   | 使用GBRT的贝叶斯优化 |
| gp_minimize     | 使用高斯过程的贝叶斯优化 |

```python
from skopt import gp_minimize

# perform optimization
result = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=100,
    random_state=42,
    verbose=True
)
```

## Step 4 评估输出

打印最佳得分和最佳参数

```python
# summarizing finding:
print(f'Best score: {result.fun}') 
print(f'Best parameters: {result.x}')
```

打印优化过程中的目标函数值

```python
print(result.func_vals)
```

绘制收敛轨迹

```python
# plot convergence traces
from skopt.plots import plot_convergence
plot_convergence(result) 
```

## Scikit-Learn API

Scikit-optimize 提供了一个类似于 GridSearchCV 和 RandomizedSearchCV 的接口 BayesSearchCV，实现了 fit 和 score 方法，以及 predict, predict_proba, decision_function, transform and inverse_transform  等常用方法。

```python
from skopt.searchcv import BayesSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# define search space 
params = {
    "max_depth": (2, 10), # integer valued parameter
    "learning_rate": (0.001, 1.0, 'log-uniform')
    "n_estimators": [100, 200, 300, 400],
    "subsample": (0.2, 0.9, 'uniform')
    "max_features": ["sqrt", "log2"],  # categorical parameter
}

# define the search
optimizer = BayesSearchCV(
    estimator=GradientBoostingRegressor(),
    search_spaces=params,
    cv=5,
    n_iter=100,
    scoring="accuracy",
    verbose=4,
    random_state=42
)

# executes bayesian optimization
optimizer.fit(X_train, y_train)

# report the best result
print(optimizer.best_score_)
print(optimizer.best_params_)
```

# Optuna

Optuna是目前为止最成熟、拓展性最强的超参数优化框架，它是专门为机器学习和深度学习所设计。为了满足机器学习开发者的需求，Optuna拥有强大且固定的API，因此Optuna代码简单，编写高度模块化，

Optuna可以无缝衔接到PyTorch、Tensorflow等深度学习框架上，也可以与sklearn的优化库scikit-optimize结合使用，因此Optuna可以被用于各种各样的优化场景。

<img src="https://warehouse-1310574346.cos.ap-shanghai.myqcloud.com/images/python/optuna-logo.png" style="zoom: 33%;" />

官方文档：https://optuna.org/

安装库

```sh
pip install optuna
```

Optuna 优化过程主要分为3步：

Step 1 构建目标函数及参数空间
Step 2 执行优化
Step 3 评估输出

## Step 1 构建目标函数及参数空间

Optuna 基于 Trial 和 Study 两个组件实现优化（optimization）。在优化过程中，Optuna 反复调用目标函数，在不同的参数下对其进行求值。一个 Trial 对应着目标函数的单次执行。在每次调用目标函数的时候，它都被内部实例化一次。而 suggest API (例如 suggest_uniform()) 在目标函数内部调用，被用于获取单个 trial 的参数。

Optuna 允许在目标函数中定义参数空间和目标，优化器会通过trail所携带的方法来构造参数空间。

| optuna.trial.Trial                                            | 说明      |
| ------------------------------------------------------------- | ------- |
| trial.suggest_categorical(name, choices)                      | 适用于分类参数 |
| trial.suggest_int(name, low, high, step=1, log=False)         | 适用于整数参数 |
| trial.suggest_float(name, low, high, *, step=None, log=False) | 适用于浮点参数 |
| trial.suggest_uniform(name, low, high)                        | 均匀分布    |
| trial.suggest_loguniform(name, low, high)                     | 对数均匀分布  |
| trial.suggest_discrete_uniform(name, low, high, q)            | 离散均匀分布  |

通过可选的 step 与 log 参数，我们可以对整形或者浮点型参数进行离散化或者取对数操作。

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# define the search space and the objecive function
def objective(trial):
    # Define the search space
    params= {
       'max_depth': trial.suggest_int('max_depth',  2, 10, 1),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 400, 100),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0, 0.1),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    }
    reg = GradientBoostingRegressor(**params)
    mse = cross_val_score(reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5).mean()
    return mse
```

通常使用交叉验证来避免过拟合。

## Step 2 执行优化

下面是几个常用术语：

- Trial: 目标函数的单次调用
- Study: 一次优化过程，包含一系列的 trials.
- Parameter: 待优化的参数

在 Optuna 中，我们用 study 对象来管理优化过程。 create_study() 方法会返回一个 study 对象 ，可以填写minimize或maximize确定优化方向，然后通过 .optimize 方法执行优化过程。

可以通过模块optuna.sampler来定义优化算法：

- GridSampler：网格搜索
- RandomSampler：随机抽样
- TPESampler：使用TPE (Tree-structured Parzen Estimator) 算法
- CmaEsSampler：使用 CMA-ES算法

```python
from optuna.samplers import TPESampler

# create a study object 
study = optuna.create_study(direction="maximize", sampler=TPESampler())

# # Invoke optimization of the objective function.
study.optimize(objective, n_trials=100)
```

获得 trial 的数目：

```python
len(study.trials)
```

Out: 100

再次执行 optimize()，可以继续优化过程

```python
study.optimize(objective, n_trials=100)
```

获得更新后的的 trial 数量：

```python
len(study.trials)
```

Out: 200

## Step 3 评估输出

打印最佳最佳分数和超参数值

```python
print(f'Best score: {study.best_value}') 
print('Best parameters: ', *[f'- {k} = {v}' for k,v in study.best_params], sep='\n')
```

Optuna 中提供了不同的方法来可视化优化结果：

| 函数                               | 说明               |
| -------------------------------- | ---------------- |
| plot_contour(study)              | 将参数关系绘制成等值线      |
| plot_intermidiate_values(study)  | 绘制所有trial的学习曲线   |
| plot_optimization_history(study) | 绘制所有trial的优化历史记录 |
| plot_param_importances(study)    | 绘制超参数重要性及其值      |
| plot_edf(study)                  | 绘制study目标值的edf   |

```python
optuna.visualization.plot_optimization_history(study)
```

## 参数空间进阶

在 Optuna 中，我们使用和 Python 语法类似的方式来定义搜索空间，其中包含条件和循环语句。

分支：

```python
import sklearn.ensemble
import sklearn.svm

def objective(trial):
    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)
```

循环：

```python
import torch
import torch.nn as nn

def create_model(trial, in_size):
    n_layers = trial.suggest_int("n_layers", 1, 3)

    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        layers.append(nn.Linear(in_size, n_units))
        layers.append(nn.ReLU())
        in_size = n_units
    layers.append(nn.Linear(in_size, 10))

    return nn.Sequential(*layers)
```

## 多目标优化

```python
from sklearn.metrics import make_scorer, root_mean_squared_error
def objective(trial):
    # Define the search space
    params= {
       'max_depth': trial.suggest_int('max_depth',  2, 10, 1),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 400, 100),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0, 0.1),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    }
    reg = GradientBoostingRegressor(**params)
    mse = cross_val_score(reg, X_train, y_train, scoring=make_scorer(root_mean_squared_error), cv=5).mean()
    r2 = cross_val_score(reg, X_train, y_train, scoring='r2', cv=5).mean()
    return mse, r2

study = optuna.create_study(directions=["minimize", "maximize"])
study.optimize(objective, n_trials=100, timeout=300)
print("Number of finished trials: ", len(study.trials))
```

> 注意：多目标优化使用的参数 directions 和单目标参数direction不同。

## 分布式优化

可使用 [Joblib Apache Spark 后端](https://github.com/joblib/joblib-spark)将 Optuna 试验分发到 Azure Databricks 群集中的多台计算机。

[使用 Optuna 进行超参数优化](https://learn.microsoft.com/zh-cn/azure/databricks/machine-learning/automl-hyperparam-tuning/optuna?source=recommendations#parallelize-optuna-trials-to-multiple-machines)

```python
import joblib
from joblibspark import register_spark

register_spark() # register Spark backend for Joblib
with joblib.parallel_backend("spark", n_jobs=-1):
    study.optimize(objective, n_trials=100)
```

## 常见问题

官方链接：https://optuna.readthedocs.io/en/stable/faq.html

### 如何定义带有额外参数的目标函数？

有两种方法可以实现这类函数。

首先，如下例所示，可调用的 objective 类具有这个功能：

```python
import optuna


class Objective:
    def __init__(self, min_x, max_x):
        # Hold this implementation specific arguments as the fields of the class.
        self.min_x = min_x
        self.max_x = max_x

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        x = trial.suggest_float("x", self.min_x, self.max_x)
        return (x - 2) ** 2


# Execute an optimization by using an `Objective` instance.
study = optuna.create_study()
study.optimize(Objective(-100, 100), n_trials=100)
```

其次，你可以用 `lambda` 或者 `functools.partial` 来创建带有额外参数的函数（闭包）。 下面是一个使用了 `lambda` 的例子：

```python
import optuna

# Objective function that takes three arguments.
def objective(trial, min_x, max_x):
    x = trial.suggest_float("x", min_x, max_x)
    return (x - 2) ** 2


# Extra arguments.
min_x = -100
max_x = 100

# Execute an optimization by using the above objective function wrapped by `lambda`.
study = optuna.create_study()
study.optimize(lambda trial: objective(trial, min_x, max_x), n_trials=100)
```

其他例子参见 [sklearn_addtitional_args.py](https://github.com/optuna/optuna/blob/master/examples/sklearn_additional_args.py) .

### 如何在目标函数中保存训练好的机器学习模型？

Optuna 会保存超参数和对应的目标函数值，但是它不会存储诸如机器学习模型或者网络权重这样的中间数据。要保存模型或者权重的话，请利用你正在使用的机器学习库提供的对应功能。

在保存模型的时候，我们推荐将 `optuna.trial.Trial.number`一同存储。这样易于之后确认对应的 trial.比如，你可以用以下方式在目标函数中保存训练好的 SVM 模型：

```python
def objective(trial):
    svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
    clf = sklearn.svm.SVC(C=svc_c)
    clf.fit(X_train, y_train)

    # Save a trained model to a file.
    with open("{}.pickle".format(trial.number), "wb") as fout:
        pickle.dump(clf, fout)
    return 1.0 - accuracy_score(y_valid, clf.predict(X_valid))


study = optuna.create_study()
study.optimize(objective, n_trials=100)

# Load the best model.
with open("{}.pickle".format(study.best_trial.number), "rb") as fin:
    best_clf = pickle.load(fin)
print(accuracy_score(y_valid, best_clf.predict(X_valid)))
```

### 在优化study时，如何避免内存不足（OOM）?

如果你运行更多trials时，导致内存增加，尝试定期运行垃圾收集器。 在回调中调用`optimize()`或调用``gc.collect()` 时，设置 `gc_after_trial=True`。

```python
def objective(trial):
    x = trial.suggest_float("x", -1.0, 1.0)
    y = trial.suggest_int("y", -5, 5)
    return x + y


study = optuna.create_study()
study.optimize(objective, n_trials=10, gc_after_trial=True)

# `gc_after_trial=True` is more or less identical to the following.
study.optimize(objective, n_trials=10, callbacks=[lambda study, trial: gc.collect()])
```

### 如何保存和恢复 study？

有两种方法可以将 study 持久化。具体采用哪种取决于你是使用内存存储 (in-memory) 还是远程数据库存储 (RDB). 通过 `pickle` 或者 `joblib`, 采用了内存存储的 study 可以和普通的 Python 对象一样被存储和加载。比如用 `joblib` 的话：

```python
study = optuna.create_study()
joblib.dump(study, "study.pkl")
```

恢复 study:

```python
study = joblib.load("study.pkl")
print("Best trial until now:")
print(" Value: ", study.best_trial.value)
print(" Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")
```

# Ray[tune]

Ray Tune 是一个标准的超参数调优工具，集成了多种参数搜索算法，并且支持分布式计算，使用方式简单。同时支持pytorch、tensorflow等训练框架，和tensorboard可视化。

官方文档：[Welcome to Ray!](https://docs.ray.io/en/latest/index.html)

首先需要安装 Ray 的 Tune 模块，可以使用以下命令：

```sh
pip install "ray[tune]"
```

Ray Tune 优化过程主要分为4步：

Step 1 定义参数空间
Step 2 定义目标函数
Step 3 执行优化
Step 4 评估输出

## Step 1 定义参数空间

我们使用字典来定义超参数空间

| ray.tune                                    | 说明                                  |
| ------------------------------------------- | ----------------------------------- |
| tune.choice(categories)                     | 分类数据采样                              |
| tune.randint(lower, upper)                  | 在区间 lower, upper 均匀采样整数             |
| tune.qrandint(lower, upper, q)              | 在区间 lower, upper 离散均匀采样整数           |
| tune.lograndint(lower, upper, base=10)      | 在区间 10^lower^,10^upper^ 对数均匀采样整数    |
| tune.qlograndint(lower, upper, q,  base=10) | 在区间 10^lower^,10^upper^ 对数离散均匀采样整数  |
| tune.uniform(lower, upper)                  | 在区间 lower, upper 均匀采样浮点数            |
| tune.quniform(lower, upper, q)              | 在区间 lower, upper离散均匀采样浮点数           |
| tune.loguniform(lower, upper, base=10)      | 在区间 10^lower^,10^upper^ 对数均匀采样浮点数   |
| tune.qloguniform(lower, upper, q, base=10)  | 在区间 10^lower^,10^upper^ 对数离散均匀采样浮点数 |
| tune.randn(mean, std)                       | 正态分布采样浮点数                           |
| tune.qrandn(mean, std, q)                   | 正态分布离散采样浮点数                         |
| tune.grid_search(values)                    | 指定网格搜索                              |
| tune.sample_from(func)                      | 配置采样函数                              |

```python
# define a search space
from ray import tune, train
space = {
    'random_state': 42, 
    'max_depth': tune.randint(2, 10),
    'learning_rate': tune.loguniform(0.001, 1.0),
    'n_estimators': tune.qrandint(20, 200, 20),
    'subsample': tune.quniform(0.1, 1.0, 0.1),
    'max_features': tune.choice(['sqrt', 'log2'])
}
```

## Step 2 定义目标函数

该函数模型训练函数（trainable）接受超参数字典，在整个训练过程结束后 `return {"score": score}` 

```python
# define an objective function
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

def trainable(params):    
    reg = GradientBoostingRegressor(**params)
    mse = cross_val_score(reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5).mean()
    return return {"score": -mse}
```

或在训练过程中的每个epoch后进行报告 `train.report({"score": score})`，便于在训练过程中监控指标的变化趋势。

```python
def objective(x, a, b):  # Define an objective function.
    return a * (x**0.5) + b

def trainable(config):  # Pass a "config" dictionary into your trainable.
    for x in range(20):  # "Train" for 20 iterations and compute intermediate scores.
        score = objective(x, config["a"], config["b"])
        train.report({"score": score})  # Send the score to Tune.
```

## Step 3 执行优化

使用 `Tuner` 创建用于调参的优化器（turner），其输入参数包括：

- `trainable`：模型训练函数

- `parame_space`：超参数搜索空间

- `tune_config`：以`tune.TuneConfig`实例作为输入，配置优化算法、度量指标等。

- `run_config`：以 `tune.RunConfig` 实例作为输入，配置训练终止条件，check point，运行结果存储路径等

然后调用方法 `.fit` 启动优化。默认情况下，Tune 会自动使用全部资源并行运行。

```python
from ray.tune.search.hyperopt import HyperOptSearch 

# Initialize 
tuner = tune.Tuner(
    objective, # Objective Function to optimize
    param_space=search_space,  # Hyperparameter's Search Space
    tune_config=tune.TuneConfig(
        metric="score",  
        mode="min",  # minimize the objective over the space
        search_alg=HyperOptSearch(),  # Optimization algorithm
        num_samples=100, # Number of optimization attempts
        time_budget_s=3600
    ),
    run_config=tune.RunConfig(
        storage_path="~/ray_results",
        verbose=1,
        stop=None
    )
)
results = tuner.fit()
```

tune 具有与许多流行的优化库集成的搜索算法，如果未指定搜索算法，将默认使用随机搜索。

| SearchAlgorithm                                                                                       | Summary                      | Website                                                                | Code Example                                                                                                |
| ----------------------------------------------------------------------------------------------------- | ---------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| [Random search/grid search](https://docs.ray.io/en/latest/tune/api/suggestion.html#tune-basicvariant) | Random search/grid search    |                                                                        | [tune_basic_example](https://docs.ray.io/en/latest/tune/examples/includes/tune_basic_example.html)          |
| [AxSearch](https://docs.ray.io/en/latest/tune/api/suggestion.html#tune-ax)                            | Bayesian/Bandit Optimization | [[Ax](https://ax.dev/)]                                                | [AX Example](https://docs.ray.io/en/latest/tune/examples/includes/ax_example.html)                          |
| [HyperOptSearch](https://docs.ray.io/en/latest/tune/api/suggestion.html#tune-hyperopt)                | Tree-Parzen Estimators       | [[HyperOpt](http://hyperopt.github.io/hyperopt)]                       | [Running Tune experiments with HyperOpt](https://docs.ray.io/en/latest/tune/examples/hyperopt_example.html) |
| [BayesOptSearch](https://docs.ray.io/en/latest/tune/api/suggestion.html#bayesopt)                     | Bayesian Optimization        | [[BayesianOptimization](https://github.com/fmfn/BayesianOptimization)] | [BayesOpt Example](https://docs.ray.io/en/latest/tune/examples/includes/bayesopt_example.html)              |
| [TuneBOHB](https://docs.ray.io/en/latest/tune/api/suggestion.html#suggest-tunebohb)                   | Bayesian Opt/HyperBand       | [[BOHB](https://github.com/automl/HpBandSter)]                         | [BOHB Example](https://docs.ray.io/en/latest/tune/examples/includes/bohb_example.html)                      |
| [NevergradSearch](https://docs.ray.io/en/latest/tune/api/suggestion.html#nevergrad)                   | Gradient-free Optimization   | [[Nevergrad](https://github.com/facebookresearch/nevergrad)]           | [Nevergrad Example](https://docs.ray.io/en/latest/tune/examples/includes/nevergrad_example.html)            |
| [OptunaSearch](https://docs.ray.io/en/latest/tune/api/suggestion.html#tune-optuna)                    | Optuna search algorithms     | [[Optuna](https://optuna.org/)]                                        | [Running Tune experiments with Optuna](https://docs.ray.io/en/latest/tune/examples/optuna_example.html)     |

## Step 4 评估输出

`Tuner.fit()`返回一个`ResultGrid`对象

```python
best_result = results.get_best_result()  # Get best result object
best_config = best_result.config  # Get best trial's hyperparameters
best_logdir = best_result.path  # Get best trial's result directory
best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
best_metrics = best_result.metrics  # Get best trial's last results
best_result_df = best_result.metrics_dataframe  # Get best result as pandas dataframe
```

此对象还可以转化为 DataFrame， 进行临时数据分析。

```python
# Get a dataframe with the last results for each trial
df_results = results.get_dataframe()

# Get a dataframe of results for a specific score or mode
df = results.get_dataframe(filter_metric="score", filter_mode="max")
```

