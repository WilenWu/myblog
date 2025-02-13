---
title: Python(Machine Learning)--CatBoost
tags:
  - Python
  - 机器学习
categories:
  - Python
  - Machine Learning
cover: /img/catboost.svg
top_img: /img/catboost-top.png
abbrlink: dc8936d5
date: 2025-02-13 00:23
description:
---

# Quick Start

Catboost 是旨在高效处理类别特征的梯度提升算法，内置多种正则化手段，减少梯度偏差和预测偏移，提高模型的准确性和泛化能力，采用对称决策树，在每个层级使用相同的特征和分割点，提升训练和预测效率。

[Jupyter Notebook Demo](/ipynb/classification_demo.html#catboost)

## Step 1:  Load the dataset

```python
from catboost import CatBoostClassifier, Pool

train_data = Pool(
    [
        [[0.1, 0.12, 0.33], [1.0, 0.7], 2, "male"],
        [[0.0, 0.8, 0.2], [1.1, 0.2], 1, "female"],
        [[0.2, 0.31, 0.1], [0.3, 0.11], 2, "female"],
        [[0.01, 0.2, 0.9], [0.62, 0.12], 1, "male"]
    ],
    label = [1, 0, 0, 1],
    cat_features=[3],
    embedding_features=[0, 1]
)

eval_data = Pool(
    [
        [[0.2, 0.1, 0.3], [1.2, 0.3], 1, "female"],
        [[0.33, 0.22, 0.4], [0.98, 0.5], 2, "female"],
        [[0.78, 0.29, 0.67], [0.76, 0.34], 2, "male"],
    ],
    label = [0, 1, 1],
    cat_features=[3],
    embedding_features=[0, 1]
)
```

catboost 可以通过 `cat_feature` 参数直接支持分类特征，而不需要重编码。

## Step 2: Setting Parameters & Training

```python
# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=10)

# train the model
model.fit(train_data, eval_set=eval_data)
```

## Step 3: Save and load model

```python
# Save model
model.save_model('catboost_model.cbm') # CatBoost binary format
# load model
loaded_model = CatBoostClassifier()
loaded_model.load_model('catboost_model.cbm')
```

Catboost 支持多种模型格式，例如保存为生产上最常用的PMML格式

```python
model.save_model('catboost_model.pmml', format="pmml")
```

## Step 4:  Predict

```python
# Get predicted classes
preds_class = model.predict(eval_data)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(eval_data)
# Get predicted RawFormulaVal
preds_raw = model.predict(eval_data, prediction_type='RawFormulaVal')
```

## Step 5:  Evaluating

```python
evals_result = model.get_evals_result()
train_logloss = evals_result['learn']['Logloss'][-1]
test_logloss = evals_result['validation']['Logloss'][-1]
```

# 参数

与 xgboost 和 lightgbm 不同，catboost 默认使用 sklearn API 训练模型。通过这个接口，跟sklearn中其他模型一样，使用fit方法进行训练，并利用predict进行结果输出。

在catboost的sklearn API中，我们可以看到下面三个类：

| module                        | comment      |
| ----------------------------- | ------------ |
| `catboost.CatBoost`           | 可实现分类和回归     |
| `catboost.CatBoostClassifier` | 实现catboost分类 |
| `catboost.CatBoostRegressor`  | 实现catboost回归 |

`CatBoost`类只有一个字典参数 `params` ，通过配置 `loss_function` 可实现分类和回归。另外两个类的参数需要像sklearn一样配置在训练模型的参数列表中。

## Pool

catboost支持numpy、pandas等经典数据结构，也可以使用自定义的数据结构Pool，这一数据结构能够保catboost算法运行更快，

| 数据结构                    | 说明   |
| ----------------------- | ---- |
| `catboost.Pool`         | 数据结构 |
| `catboost.FeaturesData` |      |

```python
catboost.Pool(
        data,
        label=None,
        cat_features=None,
        text_features=None,
        embedding_features=None,
        embedding_features_data=None,
        column_description=None,
        pairs=None,
        graph=None,
        delimiter='\t',
        has_header=False,
        ignore_csv_quoting=False,
        weight=None,
        group_id=None,
        group_weight=None,
        subgroup_id=None,
        pairs_weight=None,
        baseline=None,
        timestamp=None,
        feature_names=None,
        feature_tags=None,
        thread_count=-1,
        log_cout=sys.stdout,
        log_cerr=sys.stderr,
        data_can_be_none=False
)
```

常用参数：

- `data` 数据类型为 `list, numpy, pandas` 等
- `label` 目标变量
- `feature_names` 特征名列表
- `cat_features` 分类列索引或分类列名
- `embedding_features` embedding 特征的列索引或列名
- `column_description` 包含列描述的文件路径
- `weight` 样本权重
- `baseline` 初始值
- `log_cout, log_cerr` 日志记录

## 一般参数

- `boosting_type`：指定算法类型。
  - Ordered — 通常在小型数据集效果更好，但可能比 Plain 方案慢。
  - Plain — 经典的梯度提升方案
- `boost_from_average` 是否启用初始值 
- `classes_count` 多分类的类别数。default=None
- `thread_count`  并行的线程数。default = -1
- `task_type` 用于训练的设备。default = `cpu`, options: `cpu`, `gpu`
- `devices` 用于训练的 GPU 设备的 ID（索引从 0 开始）。default=NULL （所有的设备都被用）。格式：一个设备`devices='3'`，多个设备 `devices='0:1:3` ，一系列设备 `devices='0-3'`
- `random_state`：随机种子。别名 `random_seed`
- `logging_level` 打印日志级别。default=Verbose
  - Silent — 不输出。
  - Verbose — 打印指标、训练经过的时间和剩余训练时间
  - Info  — 输出其他信息和树的数量。
  - Debug  — 输出调试信息。
- `verbose` 打印日志配置。不能与logging_level 一起使用。default=1。别名 `verbose_eval`
- `metric_period` 计算objective和metric的频率，使用此参数可加快训练速度。default=1。

## 样本处理参数

- `scale_pos_weight` 正样本的权重，典型值：`sum(negative instances)/sum(positive instances)` 。default=1
- `class_weights` 类权重，适用于二元分类和多分类问题。 default=None
- `auto_class_weights` 自动计算类权重。支持 `None, Balanced, SqrtBalanced` ，default=None
- `class_names` 类名称。字符串格式 `smartphone,touchphone,tablet`
- `nan_mode` 处理缺失值的方法 。default=Min
  - Forbidden — 不支持缺失值
  - Min — 缺失值将作为特征的最小值处理
  - Max — 缺失值将作为特征的最大值处理

## 特征处理参数

- `one_hot_max_size` 类别数量小于阈值的特征选择one-hot编码。
- `monotone_constraints` 指定数值特征单调约束。default=None
- `feature_weights` 特征权重。default=1
- `simple_ctr` 分类特征的量化方法
- `combinations_ctr` 分类特征组合方法（特征衍生）
- `max_ctr_complexity` 允许组合的最大特征数（特征衍生）。
- `ctr_leaf_count_limit` 具有分类特征的叶的最大数量
- `target_border` 用于将目标变量转换为 0 和 1 的阈值。default=None
- `border_count` 数值特征的分割数
- `feature_border_type` 数值特征的量化模式。default=GreedyLogSum。options: Median, Uniform, UniformAndQuantiles, MaxLogSum, MinEntropy, GreedyLogSum

## 决策树生成

- `depth` 一棵树的最大深度。default=6。别名 `max_depth`
- `max_leaves` 要添加的最大节点数，只能与 Lossguide 增长策略一起使用。default=31。别名 `num_leaves`
- `min_data_in_leaf` 每个节点所需的最小实例数量。default=1。别名 `min_child_samples`
- `grow_policy` 控制将新节点添加到树中的方式。default=`SymmetricTree`
  - SymmetricTree —逐级构建树，在每次迭代中，所有叶子都使用相同的条件进行拆分，生成对称树。
  - Depthwise — 逐级构建树，在每次迭代中，每片叶子都按最优损失进行拆分。
  - Lossguide — 逐个叶子构建树，每片叶子都按最优损失进行拆分。

## 迭代过程

- `iterations` 可以构建的树的最大数量，default=1000。别名 ``num_boost_round, n_estimators, num_trees`
- `best_model_min_trees` 树的最小棵树，应与`use_best_model`一起使用。default=None
- `use_best_model`：保存到最优的迭代次数。default= True
- `learning_rate`  学习率，范围 [0,1]。default=0.03，如果未设置参数 `leaf_estimation_iterations, l2_leaf_reg` 则会根据迭代次数自动调整学习率。别名 `eta`
- `l2_leaf_reg` L2正则化系数。default=3.0，别名 `reg_lambda`
- `bootstrap_type` 权重采样方法，支持 Bayesian, Bernoulli, MVS, Poisson (supported for GPU only), No。默认值依赖于  `objective`, `task_type`, `bagging_temperature` and `sampling_unit`。
- `random_strength` 选择树结构时用于评分分割的随机性系数。default=1
- `bagging_temperature` 使用  Bayesian Bootstrap 的参数，在分类和回归模式下使用。取值范围 `(0, inf]`。default=1
- `subsample` 训练集的采样比率。在 Bernoulli, MVS, Poisson 采样方法下使用。默认值取决于数据集大小和 Bootstrap 类型。
- `sampling_frequency` 采样频率。default=`PerTreeLevel`
  - `PerTree`： 在构造每个新树之前
  - `PerTreeLevel`：在选择树的每个新拆分之前
- `sampling_unit` 抽样方案。default=`Object`
  - `Object`：The weight $w_i$​ of the i-th object $o_i$ is used for sampling the corresponding object.
  - `Group` ：The weight $w_j$ of the group $g_j$ is used for sampling each object $o_{ij}$ from the group $g_j$​.
- `rsm` 每次分裂选择时抽样，default=1。别名 `colsample_bylevel`
- `mvs_reg` 影响分母的权重，可用于平衡重要性抽样和伯努利抽样
- `model_shrink_rate` 每次迭代时的收缩系数
- `model_shrink_mode`收缩系数的计算模式 `Constant, Decreasing` 。default=Constant

## 模型训练

- [`loss_function` ](https://catboost.ai/docs/en/references/training-parameters/common#loss_function)选择需要优化的损失函数，也决定了机器学习的类别。自定义损失函数也可以设置为此参数的值。
  - CatBoostClassifier 类的默认值：若目标变量只有两个类别或设置了`target_border`参数，则损失函数为 Logloss；若目标变量有多个类别且`border_count` 参数为None，则损失函数为 MultiClass
  - CatBoost 和 CatBoostRegressor 的默认值是 RMSE。
- `eval_metric` 用于过拟合检测和选择最佳模型的指标
- `custom_metric` 训练期间输出的指标，不参与优化。default=None
- `train_dir` 用于存储训练期间生成的文件的目录。default=catboost_info
- `allow_writing_files` 允许在训练期间写入分析和快照文件。fault=True
- `save_snapshot` 启用模型快照。default=None
- `snapshot_interval` 快照保存周期。default=600s
- `snapshot_file`快照保存路径。default=experiment.cbsnapshot
- `roc_file` 在交叉验证模式下 ROC 文件输出路径。default=None

## 自定义损失函数

catboost 通过参数loss_function和eval_metric来自定义损失函数和评估函数。

自定义损失函数是一个含有calc_ders_range方法的类，接受预测变量approxes，目标变量targets和样本权重weights作为输入，返回损失函数的一阶(grad)和二阶(hess)导数。

```python
class RmseObjective(object):
    def calc_ders_range(self, approxes, targets, weights=None):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            der1 = targets[index] - approxes[index]
            der2 = -1

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result

model = CatBoostClassifier(loss_function=MultiClassObjective())
```

评估函数是一个含有is_max_optimal，evaluate，get_final_error方法的类，接受预测变量approxes，目标变量targets和样本权重weights作为输入

```python
class RmseMetric(object):
    def get_final_error(self, error, weight):
        # Returns final value of metric based on error and weight
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        # Returns whether great values of metric are better
        return False

    def evaluate(self, approxes, target, weight):
        # Returns pair (error, weights sum)
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * ((approx[i] - target[i])**2)

        return error_sum, weight_sum

model = CatBoostRegressor(eval_metric=RmseMetric())
```

## 回调参数

catboost 主要在过拟合检测中用到内置回调参数

- `early_stopping_rounds` 将过拟合检测器类型设置为 Iter，当验证集上的精度若干轮不下降，提前停止训练。 default=False
- `od_type` 要使用的过拟合检测器的类型 ，default=IncToDec。option: `IncToDec, Iter`
- `od_pval` IncToDec  检测器的阈值，当达到指定值时，训练将停止。建议 `[1e-10, 1e-2]`  。default=0 （不启用）。
- `od_wait` 在具有最佳指标值的迭代后继续训练的迭代次数。  default=20。

CatBoost 的 early_stopping_rounds 实现主要位于 C++ 核心代码中，Python 接口通过调用底层 C++ 实现来完成功能。

```python
# Built-in early stopping based on overfitting detection.
model.fit(
    train_pool,
    early_stopping_rounds=50,  # Stops if no improvement in 50 iterations
    eval_metric='Logloss'
)
```

另外在 `CatBoostClassifier` 和 `CatBoostRegressor` 构造函数中使用 `callback` ，在 `CatBoost` 的 `params` 参数中配置 `callback`，还可以在 fit 方法中使用 `callback`参数，可实现自定义回调。

然而，CatBoost 的 callback 系统设计较为基础，文档和示例不足，在设计和公开程度上远不如其他梯度提升框架（如 catboost、LightGBM）。这主要源于其架构设计、开发优先级以及用户需求的平衡。CatBoost 的回调系统设计可能受到性能优化的影响。例如，底层 C++ 核心代码对训练流程的封装较深，暴露过多接口可能引入性能开销或兼容性问题。

CatBoost 的回调接口目前仅支持 after_iteration 这一触发点，在每个训练轮次结束时被调用，接收一个`info`对象，它包含了历史指标信息：

| 字段名      | 数据类型 | 描述                                                         |
| ----------- | -------- | ------------------------------------------------------------ |
| `iteration` | `int`    | 当前迭代次数（从 1 开始）                                    |
| `metrics`   | `dict`   | 包含训练和验证指标的历史记录，键为 `'learn'` 或 `'validation'` |

```python
from catboost import CatBoostClassifier, Pool
import numpy as np

class EarlyStoppingCallback:
    def __init__(self, metric_name='Logloss', patience=5, min_delta=0.001):
        self.metric_name = metric_name
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def after_iteration(self, info):
        # Get current validation metric
        current_score = info.metrics['validation'][-1]

        # Initialize best score
        if self.best_score is None:
            self.best_score = current_score
            return True  # Continue training

        # Check for improvement
        if current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        # Stop training if no improvement for 'patience' iterations
        if self.counter >= self.patience:
            print(f"Early stopping at iteration {info.iteration}. Best {self.metric_name}: {self.best_score:.4f}")
            return False  # Stop training

        return True  # Continue training 

model.fit(
    eval_metric=['Logloss', 'AUC'],
    callbacks=[EarlyStoppingCallback(metric_name='AUC')]
) 
```

## fit 参数

```python
fit(X, y=None, 
    cat_features=None, 
    text_features=None, 
    embedding_features=None, 
    graph=None, 
    sample_weight=None, 
    baseline=None, 
    use_best_model=None, 
    eval_set=None, 
    verbose=None, 
    verbose_eval=None, 
    metric_period=None, 
    silent=None, 
    logging_level=None, 
    plot=False, 
    plot_file=None, 
    column_description=None, 
    early_stopping_rounds=None, 
    save_snapshot=None, 
    snapshot_file=None, 
    snapshot_interval=None, 
    init_model=None, 
    callbacks=None, 
    log_cout=None, 
    log_cerr=None)
```

> 某些参数与模型类构造函数中指定的参数重复。在这些情况下，fit 方法指定的值优先。

常用参数：

- `X` 数据类型为 `Pool, list, numpy, pandas` 等
- `y` 如果 `X` 配置的类型为 `Pool` 则不需要使用此参数
- `cat_features` 分类列索引或分类列名
- `embedding_features` embedding 特征的列索引或列名
- `sample_weight` 样本权重
- `baseline` 初始值
- `eval_set` 验证数据集
- `plot, plot_file` 可视化训练过程
- `early_stopping_rounds` 将过拟合检测器类型设置为 Iter，并设置早停迭代数
- `init_model` 继续训练
- `log_cout, log_cerr` 日志记录

## predict 参数

> 仅当具有特征值的参数包含模型中使用的所有特征时，模型预测结果才会正确。通常，这些功能的顺序必须与训练期间提供的相应列的顺序匹配。但是，如果在训练期间和应用模型时都提供了特征名称，则可以按名称而不是列顺序匹配它们，如 `Pool, Pandas, etc`。

```python
predict(data,
    prediction_type=None,
    ntree_start=0,
    ntree_end=0,
    thread_count=-1,
    verbose=None,
    task_type="CPU")
```

`prediction_type` 参数支持以下返回类型：

- Probability：二分类概率值
- Class：类标签
- RawFormulaVal：原始计算值
- Exponent：指数
- LogProbability：对数概率

# 训练 API

由于catboost的参数列表过长、参数类型过多，直接将所有参数混写在训练模型的类中会显得代码冗长且混乱，因此我们往往会使用字典单独呈现参数。准备好参数列表后，我们也可以使用catboost中自带的方法`xgb.train`或`xgb.cv`进行训练。

| module           | comment |
| ---------------- | ------- |
| `catboost.train` | 指定参数训练  |
| `catboost.cv`    | 交叉验证训练  |

```python
catboost.cv(pool=None, dtrain=None, 
    params=None, 
    iterations=1000, 
    num_boost_round=None,
    fold_count=3, 
    nfold=None, 
    type='Classical', 
    folds=None,
    inverted=False, 
    partition_random_seed=0, 
    seed=None,
    shuffle=True, 
    stratified=None, 
    logging_level=None, 
    metric_period=None,
    verbose=None, 
    verbose_eval=None, 
    plot=False, 
    plot_file=None, 
    early_stopping_rounds=None,
    save_snapshot=None, 
    snapshot_file=None, 
    snapshot_interval=None, 
    metric_update_interval=0.5,
    as_pandas=True, 
    return_models=False, 
    log_cout=None, 
    log_cerr=None)
```

主要参数：

- `params` - 参数字典
- `pool` 别名 `dtrain` 用于训练的数据集
- `num_boost_round` 迭代次数
- `early_stopping_rounds` 将过拟合检测器类型设置为 Iter，在指定的迭代次数后停止训练。
- `logging_level, `
- `fold_count` 别名 `nfold` 交叉验证数
- `type` 拆分方法 `Classical, Inverted, TimeSeries`，默认 Classical
- `folds` 自定义的拆分索引
- `shuffle` 拆分前随机排序
- `stratified` 是否分层抽样，分类问题默认为 True
- `as_pandas` 将返回值的类型设置为 pandas
- `return_models` 是否返回交叉验证模型列表
- `evals` 别名 `eval_set`  在 `train` 中用于验证数据集

# 可视化

Catboost 提供了可视化插件，在培训过程中和之后都可以访问此信息。必须安装`ipywidgets`软件包，才能在Jupyter Notebook中绘制图表。

```sh
pip install ipywidgets
```

启用插件

```sh
jupyter nbextension enable --py widgetsnbextension
```

通过将模型 fit 方法或 train、cv 函数的plot参数设置为True，即可在训练时绘制图表。

也可以使用类`catboost.MetricVisualizer(train_dirs, subdirs=False)`为给定目录中具有日志的所有训练、指标评估和交叉验证运行绘制指标。

# 实用程序

- `sum_models(models, weights=None, ctr_merge_policy='IntersectingCountersAverage')` 将两个或多个经过训练的 CatBoost 模型的树和计数器混合到一个新模型中。可以为每个输入模型单独加权叶值。

- `to_classifier(model)` 将 CatBoost 类型的模型转换为 CatBoostClassifier 类型的模型。

- `to_regressor(model)` 将 CatBoost 类型的模型转换为 CatBoostRegressor 类型的模型。
