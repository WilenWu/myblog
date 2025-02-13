---
title: Python(Machine Learning)--XGBoost
tags:
  - Python
  - 机器学习
categories:
  - Python
  - Machine Learning
cover: /img/XGBoost-cover.svg
top_img: /img/XGBoost-cover.svg
abbrlink: c46d5dae
date: 2024-03-28 22:42:00
description:
---

# Quick Start

XGBoost本质上还是一个GBDT，但是力争把速度和效率发挥到极致，所以叫 Extreme Gradient Boosting。XGBoost高效地实现了GBDT算法，并进行了算法和工程上的许多改进，被广泛应用在Kaggle竞赛及其他许多机器学习竞赛中，并取得了不错的成绩。

而在实际建模环节，XGBoost提供了Sklearn API和原生API两套调用方法。大部分时候我们使用原生代码来运行xgboost，因为这套原生代码是完全为集成学习所设计的，不仅可以无缝使用交叉验证、默认输出指标为RMSE，还能够默认输出训练集上的结果帮我们监控模型。

1. 首先，原生代码必须使用XGBoost自定义的数据结构DMatrix，这一数据结构能够保证xgboost算法运行更快，并且能够自然迁移到GPU上运行。
2. 当设置好数据结构后，我们需要以字典形式设置参数。XGBoost也可以接受像sklearn一样，将所有参数都写在训练所用的类当中，然而由于xgboost的参数列表过长、参数类型过多，直接将所有参数混写在训练模型的类中会显得代码冗长且混乱，因此我们往往会使用字典单独呈现参数。
3. 准备好参数列表后，我们将使用xgboost中自带的方法`xgb.train`或`xgb.cv`进行训练，训练完毕后，我们可以使用`predict`方法对结果进行预测。虽然xgboost原生代码库所使用的数据结构是DMatrix，但在预测试输出的数据结构却是普通的数组，因此可以直接使用sklearn中的评估指标，或者python编写的评估指标进行评估。


| 数据结构                                                     | 说明                   |
| ------------------------------------------------------------ | ---------------------- |
| [`xgboost.DMatrix`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#Core-Data-Structure) | XGBoost数据集         |
| `xgboost.DataIter`                                         |迭代数据|
|`xgboost.QuantileDMatrix`|直接为hist方法生成分位数数据|
| `xgboost.Booster` |  XGBoost中的返回的模型 |

```python
xgboost.DMatrix(data, 
                label=None, 
                weight=None, 
                base_margin=None, 
                missing=None, 
                silent=False, 
                feature_names=None, 
                feature_types=None, 
                nthread=None, 
                group=None, 
                qid=None, 
                label_lower_bound=None, 
                label_upper_bound=None, 
                feature_weights=None, 
                enable_categorical=False, 
                data_split_mode=DataSplitMode.ROW)
```

常用参数：

- data 内部数据集的数据源
- label 数据标签
- weight 每个样本的权重
- feature_names 特征名称
- feature_types 数据类型。如果设置 `enable_categorical=False`，字符“c”代表分类数据，字符 "q"代表数值型数据。
- feature_weights - Set feature weights for column sampling.
- enable_categorical 允许分类特征

现在，我们来简单看看原生代码是如何实现的。

[Jupyter Notebook Demo](/ipynb/classification_demo.html#XGBoost)

## Step 1:  Load the dataset

DMatrix会将特征矩阵与标签打包在同一个对象中，且一次只能转换一组数据。并且，我们无法通过索引或循环查看内部的内容，一旦数据被转换为DMatrix，就难以调用或修改了。

因此，数据预处理需要在转换为DMatrix之前做好。如果我们有划分训练集和测试集，则需要分别将训练集和测试集转换为DMatrix。

```python
# load or create your dataset
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create DMatrix for xgboost
dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

# specify validations set to watch performance
watchlist = [(dtest, "eval"), (dtrain, "train")]
```


对于表示分类要素的所有列。之后，用户可以告诉 XGBoost 启用使用分类数据进行训练。假设您正在使用 for 分类问题，请指定 参数：enable_categorical

XGBoost 可以直接支持分类特征，而不需要 one-hot 编码。传递分类数据最简单方法是使用 dataframe ，将数据类型指定为 category。

```python
# We need to specify the data type of input  as category.
X["cat_feature"].astype("category")
```
之后通过指定参数 `enable_categorical=True` 来启用分类数据进行训练
```python
# Specify `enable_categorical` to True
dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)

booster = xgb.train({"tree_method": "hist", "max_cat_to_onehot": 5}, Xy)
# Must use JSON for serialization, otherwise the information is lost
booster.save_model("categorical-model.json")
```

> Note: 在构建 DMatrix 前，先把分类特征转换成整数型。

对于其他类型的输入，例如 numpy/cupy array，我们可以通过 `feature_types` 参数设置分类特征。“q” 或 "float" 代表数值型特征，“c”代表分类特征。

```python
#  "q" is numerical feature, while "c" is categorical feature
ft = ["q", "c", "c"]
dtrain = xgb.DMatrix(X_train, y_train, feature_types=ft, enable_categorical=True)
```

## Step 2: Setting Parameters

```python
# specify parameters via map
param = {'booster': 'dart',
         'max_depth': 5, 
         'learning_rate': 0.1,
         'objective': 'binary:logistic',
         'sample_type': 'uniform',
         'normalize_type': 'tree',
         'rate_drop': 0.1,
         'skip_drop': 0.5}

# You can also specify multiple eval metrics:
params['eval_metric'] =  ['error', 'rmse']
```

## Step 3: Training

```python
# Training a model requires a parameter list and data set:
evals_result = {}
bst = xgb.train(params,
                dtrain,
                num_boost_round=20,
                evals=watchlist,
                evals_result = evals_result，
                early_stopping_rounds=10)

# Training with 5-fold CV:
xgb.cv(params, dtrain, num_boost_round=20, nfold=5)
```

不难发现，XGBoost不需要实例化，`xgb.train`函数包揽了实例化和训练的功能，一行代码解决所有问题。
## Step 4: Save and load model

```python
# Save model
bst.save_model("model.txt")
# load model
bst = xgb.Booster(model_file="model.txt")

# alternatively, you can pickle the booster
import pickle
with open('model.pkl', 'wb') as fout:
    pickle.dump(bst, fout)
with open('model.pkl', 'rb') as fin:
    bst = pickle.load(fin)
```

XGBoost在Booster对象中有一个名为`dump_model`的函数，它允许以`text`、`json`或`dot`（graphviz）等可读格式导出模型。它的主要用于模型解释或可视化，不应该重新加载回XGBoost。
```python
# dump model
bst.dump_model("dump.raw.txt")
# dump model with feature map
bst.dump_model("dump.nice.txt", os.path.join(DEMO_DIR, "data/featmap.txt"))
```

## Step 5:  Predict

```python
# run prediction
y_pred = bst.predict(dtest)
y_true = dtest.get_label()

# If early stopping is enabled during training, you can get predictions from the best iteration with bst.best_iteration:
y_pred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
```

## Step 6:  Evaluating

```python
from sklearn.metric import mean_squared_error
rmse_test = mean_squared_error(y_true, y_pred) ** 0.5
print(f'The RMSE of prediction is: {rmse_test}')
```

# 参数

xgboost将参数分为了两大部分，一部分可以通过params进行设置，另一部分则需要在方法`xgb.train`或者`xgb.cv`中进行设置。遗憾的是，xgboost并没有明确对参数分割的条件和理由，但一般来说，除了迭代次数和提前停止这两个关键元素，其他参数基本都被设置在`params`当中。如果在实际运行过程中，出现了警告或报错，则根据实际情况进行调整。

## 一般参数

- `booster`：指定算法类型。default= `gbtree` 可以是gbtree、gblinear或dart。gbtree和dart使用基于树的模型，而gblinear使用线性函数。
- `device`：学习设备。default = cpu
- `nthread`：如果未设置，默认为最大可用线程数。
- `seed`：随机种子。
- `verbosity`：打印消息的详细性。有效值为0（静音）、1（警告）、2（信息）和3（调试），default=1。可以使用 `xgboost.config_context()` 在全局范围内设置。
-  `num_parallel_tree` 每次迭代期间构建的并行树的数量。此选项用于支持增强的随机森林。default=1
- `multi_strategy` 用于训练多目标模型的策略，包括多目标回归和多类分类。default= `one_output_per_tree`
    - `one_output_per_tree`：每个目标一个模型。
    - `multi_output_tree`：使用多目标树。

## 样本处理参数

- `scale_pos_weight` 控制正负样本的权重平衡，典型值：`sum(negative instances)/sum(positive instances)` 。default=1

## 特征处理参数

- `max_cat_to_onehot` 决定XGBoost是否使用one-hot编码的阈值。当类别数量小于阈值时，则选择one-hot编码，否则类别将被划分为子节点。
- `max_cat_threshold`  每一次分裂的最大类别数
- `max_bin` 最大分箱数。仅当`tree_method`设置为`hist`或`approx`时使用。default=256
Monotonic Constraints 单调约束
Feature Interaction Constraints 特征交互约束
## 决策树生成

- `tree_method` XGBoost中使用的树构造算法。default=`auto`
    - `auto`：与`hist`树方法相同。
    - `exact`：精确的贪婪算法。列举所有拆分候选。
    - `approx`：使用分位数草图和梯度直方图的近似贪婪算法。
    - `hist`：直方图算法。
    - 对于`refresh`等其他更新程序，请直接设置参数`updater`。
- `max_depth` 一棵树的最大深度。default=6
- `max_leaves` 要添加的最大节点数。default=0
- `min_child_weight` 每个节点所需的最小实例数量。default=1
- `grow_policy` 控制将新节点添加到树中的方式，目前仅在`tree_method`设置为`hist`或`approx`时支持。。default=`depthwise`
    - `depthwise`：在最靠近根的节点上拆分。
    - `lossguide`：在损失变化最大的节点上拆分。

## 迭代过程

- `eta`  学习率，范围 [0,1]。default=0.3，别名 `learning_rate`
- `lambda` L2正则化系数。default=1，别名 `reg_lambda`
- `alpha` L1正则化系数。default=0，别名：`reg_alpha`
- `gamma` 依照叶子总量对目标函数施加惩罚的系数。default=0，别名：`min_split_loss`

目标函数

$$
Obj_k = \sum_{i}l(y_i,\hat{y_i}) + \gamma T + \frac{1}{2}\lambda\sum_{j=1}^Tw_j^2 + \alpha\sum_{j=1}^Tw_j
$$

其中$T$表示当前第$k$棵树上的叶子总量，$w_j$则代表当前树上第$j$片叶子的叶子权重（leaf weights），即当前叶子$j$的预测值。正则项有两个：使用平方的 $\ell_2$正则项与使用绝对值的 $\ell_1$正则项。

- `subsample` 训练集的采样比率。子采样将在每次提升迭代中发生一次。default=1
- `sampling_method` 训练集的子采样方法。default=`uniform`
  - `uniform`：每个训练实例被选中的概率相等。通常设置`subsample`>= 0.5以获得良好的结果。
  - `gradient_based`：每个训练实例的选择概率与梯度的*正则化绝对值*成正比。更具体地说，`subsample`可以设置为低至0.1，而不会损失模型准确性。请注意，仅当`tree_method`设置为`hist`且设备为`cuda`才支持此采样方法；其他树方法仅支持`uniform`采样。
- `colsample_bytree`, `colsample_bylevel`, `colsample_bynode` 这是用于特征子采样的参数系列。default=1
  - `colsample_bytree`是构建每棵树时列的子采样比率。每构建一棵树，子采样都会发生一次。
  - `colsample_bylevel`是每个级别的列的子采样比率。每在树上达到一个新的深度水平，就会进行子采样。是从当前树的列集进行子采样的。
  - `colsample_bynode`是每个节点（拆分）的列子采样比率。每次评估新的拆分时，子采样都会发生一次。是从当前级别的列集进行子采样的。
  - `colsample_by*`参数累积工作。例如，具有64个特征的组合`{'colsample_bytree':0.5,'colsample_bylevel':0.5,'colsample_bynode':0.5}`将在每次拆分时留下8个特征可供选择。
  - 构建数据集时，可以为DMatrix设置`feature_weights`，以定义使用列采样时选择每个特征的概率。sklearn界面中的`fit`方法有一个类似的参数。
- `updater` 提供了构建和修改树的模块化方式。这是一个高级参数，通常会自动设置。
    - `grow_colmaker`：非分布式柱式树结构。
    - `grow_histmaker`：基于直方图计数的全局提案，基于行的数据拆分的分布式树结构。
    - `grow_quantile_histmaker`：使用量化直方图构建树。
    - `grow_gpu_hist`：当`tree_method`与`device=cuda`一起设置为`hist`启用。
    - `grow_gpu_approx`：当`tree_method`与`device=cuda`一起设置为`approx`启用。
    - `sync`：同步所有分布式节点中的树。
    - `refresh`：根据当前数据刷新树的叶节点权重和/或叶节点值。请注意，不会对数据行进行随机子采样。
    - `prune`：修剪损失小于 `min_split_loss`（或`gamma`）和深度大于`max_depth`的节点。
- `refresh_leaf` 这是`refresh`更新程序的一个参数。当设置为1时，叶节点值和权重都会更新。当设置为0时，只更新叶节点权重。default=1
- `process_type` 指定提升过程。default=`default`
    - `default`：创造新树的正常提升过程。
    - `update`：从现有模型开始，仅更新现有的树。在每次提升迭代中，从初始模型中获取一棵树，为该树运行指定的更新程序，并将修改后的树添加到新模型中。新模型将具有相同或更少的树，这取决于执行的增强迭代次数。目前，仅 `updater` 设置为 `refresh`或`prune`时有意义。使用`process_type=update`时，不能使用创建新树的更新程序。
- `max_delta_step` 一次迭代中所允许的最大迭代值。通常不需要这个参数，但当类极度不平衡时，它可能有助于逻辑回归。将其设置为1-10的值可能有助于控制更新。default=0

**树方法**：对于训练 boosted tree 模型，有2个参数用于选择算法，即updater和tree_method。XGBoost有3种内置树方法，即exact、approx和hist。除了这些树方法外，还有一些独立的更新程序，包括refresh、prune和sync。参数updater比tree_method更原始，因为后者只是前者的预配置，差异主要是由于历史原因。

## 模型训练

- `objective` 选择需要优化的损失函数。default=`reg:squarederror`
  - 回归问题： `reg:squarederror`、 `reg:squaredlogerror`、 `reg:pseudohubererror`、`reg:absoluteerror`、`reg:quantileerror`、`pinballloss`、`count:poisson`、`reg:gamma`、`reg:tweedie`
  - 分类问题： `reg:logistic`、 `binary:logistic`、 `binary:logitraw`、`binary:hinge`、`multi:softprob`、`multi:softmax`
  - 生存分析：`survival:cox`、 `survival:aft`
  - 排序问题：`rank:ndcg`、 `rank:map`、`rank:pairwise`
- `base_score` 初始化预测结果$H_0$的设置
- `eval_metric` 评估指标。将根据 `objective`分配默认值：回归的rmse，分类的logloss，`rank:map`的平均精度等。支持添加多个评估指标。
  - 回归问题： `rmse`、`rmsle`、`mae`、 `mape`、`mphe`
  - 分类问题：`logloss`、 `error`、`error@t` 可以通过 t 来指定与0.5不同的二分类阈值、 `merror`、 `mlogloss`、 `auc`、`aucpr`、 `map`
- `disable_default_eval_metric`：是否禁用默认评估函数。default= False 

| module                                       | comment                        |
| --------------------- | ----------------------------- |
|xgboost.train|指定参数训练|
|xgboost.cv|交叉验证训练|


主要参数：
- params - Booster 参数字典
- dtrain 用于训练的数据集
- num_boost_round 提升迭代次数，即生成的基学习器的数量
- evals 验证/测试数据
- obj 自定义目标函数
- feval 自定义评估函数（已废弃）
- maximize 是否最大化 feval
- early_stopping_rounds 提前停止，需要至少一个evals。如果evals、eval_metric不止一个，则选用最后一个判断提前停止
- evals_result 记录验证集的评估结果 
- verbose_eval
- xgb_model 初始化模型，允许继续训练
- callbacks 回调函数列表
- custom_metric 自定义评估函数
- nfold CV值
- stratified 是否分层抽样
- folds sklearn  - a KFold or StratifiedKFold instance or list of fold indices
- as_pandas 是否转化为pandas
- show_stdv 是否打印标准差

## 回调参数

|方法	|Create a callback|
|---|---|
|`xgboost.callback.TrainingCallback`|Interface for training callback.|
|`xgboost.callback.EvaluationMonitor(rank=0, period=1, show_stdv=False)`|输出评估结果的频率|
|`xgboost.callback.EarlyStopping(rounds)`|回调提前停止策略，控制过拟合风险，当验证集上的精度若干轮不下降，提前停止训练。|
|`xgboost.callback.LearningRateScheduler(learning_rates)`|调度学习率|


```python
D_train = xgb.DMatrix(X_train, y_train)
D_valid = xgb.DMatrix(X_valid, y_valid)

# Define a custom evaluation metric used for early stopping.
def eval_error_metric(predt, dtrain: xgb.DMatrix):
    label = dtrain.get_label()
    r = np.zeros(predt.shape)
    gt = predt > 0.5
    r[gt] = 1 - label[gt]
    le = predt <= 0.5
    r[le] = label[le]
    return 'CustomErr', np.sum(r)

# Specify which dataset and which metric should be used for early stopping.
early_stop = xgb.callback.EarlyStopping(rounds=early_stopping_rounds,
                                        metric_name='CustomErr',
                                        data_name='Train')

booster = xgb.train(
    params = {'objective': 'binary:logistic',
              'eval_metric': ['error', 'rmse'],
              'tree_method': 'hist'}, 
    dtrain = D_train,
    evals=[(D_train, 'Train'), (D_valid, 'Valid')],
    feval=eval_error_metric,
    num_boost_round=1000,
    callbacks=[early_stop],
    verbose_eval=False)
```

## 自定义回调函数

XGBoost 提供了 `TrainingCallback` 基类，用于创建自定义的回调函数。回调函数可以在训练过程中的特定事件发生时被调用，例如在每个迭代（boosting round）之后。通过继承这个基类，你可以实现自定义的逻辑，比如监控训练进度、调整超参数、保存模型等。

**主要方法**

- `before_training(self, model)`：在训练开始前调用，参数`model` 是当前的 `Booster` 模型对象。返回经过可能修改的 `model` 对象。
- `after_training(self, model)`：在整个训练过程结束后调用，参数`model` 是经过训练后的 Booster 模型对象。返回经过可能修改的 `model` 对象。
- `before_iteration(self, model, epoch, evals_log)`：每一轮迭代开始前调用，返回布尔值。
- `after_iteration(self, model, epoch, evals_log)`：每一轮迭代结束后调用，返回值布尔值。如果返回 `True`，则会提前终止训练；否则继续训练。
  - `model`: 当前的 `Booster` 模型对象。
  - `epoch`: 当前迭代次数（从0开始计数）。
  - `evals_log`: 包含评估历史的日志字典。键是数据集名称，值是另一个字典，后者包含度量标准名称和对应的度量值列表 `{"data_name": {"metric_name": [0.5, ...]}}`。

```python
from xgboost.callback import TrainingCallback

class CustomEarlyStopping(TrainingCallback):
    def __init__(self, rounds):
        self.rounds = rounds
        self.best_loss = float('inf')
        self.best_iteration = 0

    def after_iteration(self, model, epoch, evals_log):
        # 获取当前轮的训练损失
        current_loss = evals_log['validation']['rmse'][epoch]
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_iteration = epoch
            print(f"Round {epoch}: Best iteration = {self.best_iteration}, Best loss = {self.best_loss:.4f}")
        if epoch - self.best_iteration >= self.rounds:
            print(f'Early stopping at epoch {epoch}')
            model.set_attr(best_iteration=str(self.best_iteration))
            return True  # Stop training
        return False
```

上述代码中，`CustomEarlyStopping`类实现了根据损失值的变化来决定是否提前停止训练的功能。在`after_iteration`方法中，比较当前轮的损失值和上一轮的损失值，如果连续 rounds 轮损失值没有下降，则返回`True`，表示提前停止训练

## 自定义损失函数

xgboost 在 xgb.train中通过参数obj和custom_metric来自定损失函数和评估函数。

自定义损失函数接受predt和dtrain作为输入，返回损失函数的一阶(grad)和二阶(hess)导数。

```python
import xgboost as xgb
from typing import Tuple

def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    y = dtrain.get_label()
    return (np.log1p(predt) - np.log1p(y)) / (predt + 1)

def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the hessian for squared log error.'''
    y = dtrain.get_label()
    return ((-np.log1p(predt) + np.log1p(y) + 1) /
            np.power(predt + 1, 2))

def squared_log(predt: np.ndarray,
                dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    
    :math:`\frac{1}{2}[log(pred + 1) - log(label + 1)]^2`
    '''
    predt[predt < -1] = -1 + 1e-6
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess
```

自定义损失函数后，模型的输出不在是 [0,1] 概率输出，而是 sigmoid 函数之前的输入值。因此，需要写出对应的评估函数。
```python
def rmsle(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    ''' Root mean squared log error metric.'''
    y = dtrain.get_label()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y) - np.log1p(predt), 2)
    return 'PyRMSLE', float(np.sqrt(np.sum(elements) / len(y)))
```
评估函数也接受predt和dtrain作为输入，返回本身的名称和浮点值作为结果。

```python
xgb.train({'tree_method': 'hist', 'seed': 1994,
           'disable_default_eval_metric': 1},
          dtrain=dtrain,
          num_boost_round=10,
          obj=squared_log,
          custom_metric=rmsle,
          evals=[(dtrain, 'dtrain'), (dtest, 'dtest')],
          evals_result=results)
```
> 请注意，参数disable_default_eval_metric用于抑制XGBoost中的默认度量。

当自定义损失函数后，模型 `predict` 将会输出原始值，需要手动进行sigmoid函数变换。可以通过`predict`函数中的`output_margin`参数来控制
# Scikit-Learn API

XGBoost的原生代码与我们已经习惯了的sklearn代码有很大的不同。对于熟悉sklearn的我们来说，许多人也会倾向于使用xgboost自带的sklearn接口来实现算法。通过这个接口，我们可以使用跟sklearn代码一样的方式来实现xgboost，即可以通过fit和predict等接口来执行训练预测过程，也可以调用属性比如coef_等。

在XGBoost的sklearn API中，我们可以看到下面五个类：

| module                                       | comment                        |
| --------------------- | ----------------------------- |
| XGBRegressor    | 实现xgboost回归               |
| XGBClassifier   | 实现xgboost分类               |
| XGBRanker      | 实现xgboost排序               |
| XGBRFClassifier | 基于xgboost库实现随机森林分类 |
| XGBRFRegressor  | 基于xgboost库实现随机森林回归 |

其中XGBRF的两个类是以XGBoost方式建树、但以bagging方式构建森林的类，通常只有在我们使用普通随机森林效果不佳、但又不希望使用Boosting的时候使用。这种使用XGBoost方式建树的森林在sklearn中已经开始了实验，不过还没有正式上线。

另外两个类就很容易理解了，一个是XGBoost的回归，一个是XGBoost的分类。这两个类的参数高度相似，我们可以以XGBoost分类为例查看：

```python
XGBClassifier(
    n_estimators : int = None,
    max_depth : int = None,
    max_leaves : int = None,
    max_bin : int = None,
    grow_policy : {0, 1} = None,
    learning_rate : float = None,
    objective: Union[str, Callable, NoneType] = "binary:logistic",
    booster : str = None, # gbtree, gblinear or dart.
    tree_method : str = None,
    n_jobs : int = None,
    gamma : float = None,
    min_child_weight : float = None,
    max_delta_step : float = None,
    subsample : float = None,
    sampling_method : {"uniform", "gradient_based"} = None,
    colsample_bytree : float = None,
    colsample_bylevel : float = None,
    colsample_bynode : float = None,
    reg_alpha : float = None,
    reg_lambda : float = None,
    scale_pos_weight : float = None,
    base_score : NoneType = None,
    missing : float = np.nan,
    num_parallel_tree : int = None,
    monotone_constraints : Union[Dict[str, int], str] = None,
    interaction_constraints : Union[str, List[Tuple[str]]] = None, 
    importance_type : str = None,
    device : {"cpu", "cuda", "gpu"} = None,
    validate_parameters : bool = None,
    enable_categorical : bool = False,
    feature_types : FeatureTypes = None,
    max_cat_to_onehot : int = None,
    max_cat_threshold : int = None,
    multi_strategy : {"one_output_per_tree", "multi_output_tree"} = None,
    eval_metric : Union[str, List[str], Callable] = None,
    early_stopping_rounds : int = None,
    callbacks : List[TrainingCallback] = None,
    random_state : Union[numpy.random.RandomState, int] = None,
    verbosity : int = None,
    **kwargs : dict = None
)
```


具体的模型训练过程和sklearn中其他模型一样，通过fit进行训练，并利用predict进行结果输出：

```python
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# read data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
# create model instance
clf = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# fit model
clf.fit(X_train, y_train)
# make predictions
preds = clf.predict(X_test)
# Save model into JSON format.
clf.save_model("clf.json")
```

# 可视化


| module                   | comment                |
| :----------------------- | :--------------------- |
| plot_importance(booster) | 绘制模型的特征重要性。 |
| plot_tree(booster)       | 绘制指定的树           |
| to_graphviz(booster)     | 创建指定树的二叉图文件 |

# 继续训练

XGBoost提供两种增量学习的方式：
- 一种是在当前迭代树的基础上增加新树，原树不变；
- 一种是当前迭代树结构不变，重新计算叶节点权重和/或叶节点值。

[Jupyter notebook 增量学习Demo](/ipynb/incremental_learning_demo.html#XGBoost)

在初始化模型 `xgb_model` 上继续训练

```python
# Train 128 iterations, with the first one runs for 32 iterations and
# the second one runs for 96 iterations
clf1 = xgboost.XGBClassifier(n_estimators=32)
clf1.fit(X, y, eval_set=[(X, y)], eval_metric="logloss")
assert clf1.get_booster().num_boosted_rounds() == 32

clf2 = xgboost.XGBClassifier(n_estimators=128 - 32)
clf2.fit(X, y, eval_set=[(X, y)], eval_metric="logloss", xgb_model=clf1)

print("Total boosted rounds:", clf.get_booster().num_boosted_rounds())
```

使用`process_type`参数更新叶节点
```python
# using `process_type` with `prune` and `refresh`
n_rounds=32

# Train a model first
Xy = xgb.DMatrix(X_train, y_train)
evals_result = {}
booster = xgb.train(
    {"tree_method": "hist", "max_depth": 6, "device": "cuda"},
    Xy,
    num_boost_round=n_rounds,
    evals=[(Xy, "Train")],
    evals_result=evals_result,
)

# Refresh the leaf value and tree statistic
Xy_refresh = xgb.DMatrix(X_refresh, y_refresh)
# The model will adapt to new data by changing leaf value (no change in
# split condition) with refresh_leaf set to True.
refresh_result = {}
refreshed = xgb.train(
    {"process_type": "update", "updater": "refresh", "refresh_leaf": True},
    Xy_refresh,
    num_boost_round=n_rounds,
    xgb_model=booster,
    evals=[(Xy, "Original"), (Xy_refresh, "Train")],
    evals_result=refresh_result,
)

# Refresh the model without changing the leaf value, but tree statistic including
# cover and weight are refreshed.
refresh_result = {}
refreshed = xgb.train(
    {"process_type": "update", "updater": "refresh", "refresh_leaf": False},
    Xy_refresh,
    num_boost_round=n_rounds,
    xgb_model=booster,
    evals=[(Xy, "Original"), (Xy_refresh, "Train")],
    evals_result=refresh_result,
)

# Prune the trees with smaller max_depth
Xy_update = xgb.DMatrix(X_update, y_update)
prune_result = {}
pruned = xgb.train(
    {"process_type": "update", "updater": "prune", "max_depth": 2},
    Xy_update,
    num_boost_round=n_rounds,
    xgb_model=booster,
    evals=[(Xy, "Original"), (Xy_update, "Train")],
    evals_result=prune_result,
)
```

# 分布式学习

> 从1.7.0版本开始，xgboost已经封装了pyspark API，因此不需要纠结spark版本对应的jar包 xgboost4j 和 xgboost4j-spark 的下载问题了，也不需要下载调度包 sparkxgb.zip。

| 算法               | 说明                                                         |
| ------------------ | ------------------------------------------------------------ |
|[xgboost.spark.SparkXGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.spark)|PySpark分类算法|
|xgboost.spark.SparkXGBRegressor|PySpark回归算法|
|xgboost.spark.SparkXGBRanker|PySpark排名算法|

[Jupyter notebook 分布式学习Demo](/ipynb/distributed_learning_demo.html#XGBoost-with-spark)

以 SparkXGBClassifier 为例，介绍下XGBoost在spark中的用法

```python
xgboost.spark.SparkXGBClassifier(
    features_col='features', 
    label_col='label', 
    prediction_col='prediction', 
    probability_col='probability', 
    raw_prediction_col='rawPrediction', 
    pred_contrib_col=None, 
    validation_indicator_col=None, 
    weight_col=None, 
    base_margin_col=None, 
    num_workers=1, 
    use_gpu=None, 
    device=None, 
    force_repartition=False, 
    repartition_random_shuffle=False, 
    enable_sparse_data_optim=False, 
    **kwargs)
```


```python
from xgboost.spark import SparkXGBClassifier
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.master("local[*]").getOrCreate()

# create dataset
df_train = spark.createDataFrame([
    (Vectors.dense(1.0, 2.0, 3.0), 0, False, 1.0),
    (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, False, 2.0),
    (Vectors.dense(4.0, 5.0, 6.0), 0, True, 1.0),
    (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, True, 2.0),
], ["features", "label", "isVal", "weight"])
df_test = spark.createDataFrame([
    (Vectors.dense(1.0, 2.0, 3.0), ),
], ["features"])

# train xgboost classifier model
clf = SparkXGBClassifier(max_depth=5, missing=0.0,
    validation_indicator_col='isVal', weight_col='weight',
    early_stopping_rounds=1, eval_metric='logloss')
model = xgb_classifier.fit(df_train)
predict_df = model.transform(df_test)

classifier_evaluator = MulticlassClassificationEvaluator(metricName="f1")
print(f"classifier f1={classifier_evaluator.evaluate(predict_df)}")
```

# 分布式预测

当我们训练好一个本地模型，想在大规模的数据上预测时，可以使用pandas_udf进行分布式预测：

```python
from pyspark.sql.functions import pandas_udf, struct
import xgboost as xgb
import pandas as pd

def predict_with_spark(spark_df, spark_context, local_model):
   var = spark_context.broadcast(local_model)
   model = var.value

   @pandas_udf('float')
   def transform(X):
      categorical = [var for var in X.columns if X[var].dtype == 'object']
      if len(categorical) > 0:
         X[categorical] = X[categorical].astype('category')
      
      X = xgb.DMatrix(X, enable_categorical=True)
      return pd.Series(model.predict(X))

   cols = struct(*model.feature_names)
   return spark_df.withColumn('predictions', transform(cols))

bst = xgb.Booster(model_file='bst.txt')
df = spark.sql("select * from home_credit_default_risk")
predict_with_spark(df, sc, bst).select('predictions').show()
```

