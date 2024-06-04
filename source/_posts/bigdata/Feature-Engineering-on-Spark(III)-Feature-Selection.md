---
title: PySpark 特征工程(III)--特征选择
tags:
  - Python
  - Spark
categories:
  - Big Data
  - Spark
cover: /img/apache-spark-mllib.png
top_img: /img/apache-spark-top-img.svg
abbrlink: d099726d
description: 
date: 2024-06-03 23:40:02
---

有这么一句话在业界广泛流传：数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。由此可见，特征工程在机器学习中占有相当重要的地位。在实际应用当中，可以说特征工程是机器学习成功的关键。

特征工程是数据分析中最耗时间和精力的一部分工作，它不像算法和模型那样是确定的步骤，更多是工程上的经验和权衡。因此没有统一的方法。这里只是对一些常用的方法做一个总结。

特征工程包含了 Data PreProcessing（数据预处理）、Feature Extraction（特征提取）、Feature Selection（特征选择）和 Feature construction（特征构造）等子问题。

Jupyter Notebook 代码连接：[feature_engineering_on_spark_p3_feature_selection](/ipynb/feature_engineering_on_spark_p3_feature_selection)

# 特征选择

现在我们已经有大量的特征可使用，有的特征携带的信息丰富，有的特征携带的信息有重叠，有的特征则属于无关特征，尽管在拟合一个模型之前很难说哪些特征是重要的，但如果所有特征不经筛选地全部作为训练特征，经常会出现维度灾难问题，甚至会降低模型的泛化性能（因为较无益的特征会淹没那些更重要的特征）。因此，我们需要进行特征筛选，排除无效/冗余的特征，把有用的特征挑选出来作为模型的训练数据。

特征选择方法有很多，一般分为三类：

- 过滤法（Filter）比较简单，它按照特征的发散性或者相关性指标对各个特征进行评分，设定评分阈值或者待选择阈值的个数，选择合适特征。
- 包装法（Wrapper）根据目标函数，通常是预测效果评分，每次选择部分特征，或者排除部分特征。
- 嵌入法（Embedded）则稍微复杂一点，它先使用选择的算法进行训练，得到各个特征的权重，根据权重从大到小来选择特征。


```python
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml import Estimator, Transformer
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
import pyspark.sql.functions as fn
import pyspark.ml.feature as ft
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.sql import Observation
from pyspark.sql import Window
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from xgboost.spark import SparkXGBClassifier
import xgboost as xgb

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
import warnings
import gc

# Setting configuration.
warnings.filterwarnings('ignore')
SEED = 42

# Use 0.11.4-spark3.3 version for Spark3.3 and 1.0.2 version for Spark3.4
spark = SparkSession.builder \
            .master("local[*]") \
            .appName("XGBoost with PySpark") \
            .config("spark.driver.memory", "10g") \
            .config("spark.driver.cores", "2") \
            .config("spark.executor.memory", "10g") \
            .config("spark.executor.cores", "2") \
            .enableHiveSupport() \
            .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('ERROR')
```

    24/06/03 21:40:26 WARN Utils: Your hostname, MacBook-Air resolves to a loopback address: 127.0.0.1; using 192.168.1.5 instead (on interface en0)
    24/06/03 21:40:26 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    24/06/03 21:40:26 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable


定义数据集评估函数


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
        print("Starting time\t", time.strftime("%H:%M:%S", time.localtime()))
        result = func(*args, **kwargs)
        delta = strfdelta(time.time() - click, "{:.0f} hours {:.0f} minutes {:.0f} seconds")
        print(f"{func.__name__} cost {delta}")
        return result
    return wrapper

def progress(percent=0, width=50, desc="Processing"):
    import math
    tags = math.ceil(width * percent) * "#"
    print(f"\r{desc}: [{tags:-<{width}}]{percent:.1%}", end="", flush=True)

def cross_val_score(df, estimator, evaluator, features, numFolds=3, seed=SEED):
    df = df.withColumn('fold', (fn.rand(seed) * numFolds).cast('int'))
    eval_result = []
    # Initialize an empty dataframe to hold feature importances
    feature_importances = pd.DataFrame(index=features)
    for i in range(numFolds):
        train = df.filter(df['fold'] == i)
        valid = df.filter(df['fold'] != i)
        model = estimator.fit(train)
        train_pred = model.transform(train)
        valid_pred = model.transform(valid)
        train_score = evaluator.evaluate(train_pred)
        valid_score = evaluator.evaluate(valid_pred)
        metric = evaluator.getMetricName()
        print(f"[{i}] train's {metric}: {train_score},  valid's {metric}: {valid_score}")
        eval_result.append(valid_score)
        
        fscore = model.get_feature_importances()
        fscore = {name:fscore.get(f'f{k}', 0) for k,name in enumerate(features)}
        feature_importances[f'cv_{i}'] = fscore
    feature_importances['fscore'] = feature_importances.mean(axis=1)
    return eval_result, feature_importances.sort_values('fscore', ascending=False)

@timer
def score_dataset(df, inputCols=None, featuresCol=None, labelCol='label', nfold=3):
    assert inputCols is not None or featuresCol is not None
    if featuresCol is None:
        # Assemble the feature columns into a single vector column
        featuresCol = "features"
        assembler = VectorAssembler(
            inputCols=inputCols,
            outputCol=featuresCol
        )
        df = assembler.transform(df)
    # Create an Estimator.
    classifier = SparkXGBClassifier(
        features_col=featuresCol, 
        label_col=labelCol,
        eval_metric='auc',
        scale_pos_weight=11,
        learning_rate=0.015,
        max_depth=8,
        subsample=1.0,
        colsample_bytree=0.35,
        reg_alpha=65,
        reg_lambda=15,
        n_estimators=500,
        verbosity=0
    ) 
    evaluator = BinaryClassificationEvaluator(labelCol=labelCol, metricName='areaUnderROC')
    # Training with 3-fold CV:
    scores, feature_importances = cross_val_score(
        df=df,
        estimator=classifier, 
        evaluator=evaluator,
        features=inputCols,
        numFolds=nfold
    )
    print(f"cv_agg's valid auc: {np.mean(scores):.4f} +/- {np.std(scores):.5f}")
    return feature_importances
```


```python
df = spark.sql("select * from home_credit_default_risk.created_data")
```

    Loading class `com.mysql.jdbc.Driver'. This is deprecated. The new driver class is `com.mysql.cj.jdbc.Driver'. The driver is automatically registered via the SPI and manual loading of the driver class is generally unnecessary.

```python
# Persists the data in the disk by specifying the storage level.
from pyspark.storagelevel import StorageLevel
_ = df.persist(StorageLevel.MEMORY_AND_DISK)
```


```python
features = df.drop('SK_ID_CURR', 'label').columns
feature_importances = score_dataset(df, inputCols=features)
```

    Starting time	 21:40:31
    [0] train's areaUnderROC: 0.8790646375204176,  valid's areaUnderROC: 0.7621647570277277
    [1] train's areaUnderROC: 0.8746030416668324,  valid's areaUnderROC: 0.7576869026346968
    [2] train's areaUnderROC: 0.8784984656392806,  valid's areaUnderROC: 0.7583365874350807
    cv_agg's valid auc: 0.7594 +/- 0.00198
    score_dataset cost 0 hours 7 minutes 33 seconds    

## 单变量特征选择

Relief（Relevant Features）是著名的过滤式特征选择方法。该方法假设特征子集的重要性是由子集中的每个特征所对应的相关统计量分量之和所决定的。所以只需要选择前k个大的相关统计量对应的特征，或者大于某个阈值的相关统计量对应的特征即可。


| pyspark.ml.feature                          |                                |
| ------------------------------------------- | ------------------------------ |
| ChiSqSelector(numTopFeatures, ...)          | 选择用于预测分类标签的分类特征 |
| VarianceThresholdSelector(featuresCol, ...) | 删除所有低方差特征             |
| UnivariateFeatureSelector(featuresCol, ...) | 单变量特征选择                 |

`UnivariateFeatureSelector`在具有分类/连续特征的分类/回归任务上选择特征。Spark根据指定的`featureType`和`labelType`参数选择要使用的评分函数。

| featureType | labelType   | score function         |
| :---------- | :---------- | :--------------------- |
| categorical | categorical | chi-squared (chi2)     |
| continuous  | categorical | ANOVATest (f_classif)  |
| continuous  | continuous  | F-value (f_regression) |

它支持五种选择模式：

- numTopFeatures 选择评分最高的固定数量的特征。
- percentile 选择评分最高的固定百分比的特征。
- fpr选择p值低于阈值的所有特征，从而控制假阳性选择率。
- fdr使用Benjamini-Hochberg程序来选择错误发现率低于阈值的所有特征。
- fwe选择p值低于阈值的所有功能。阈值按1/numFeatures缩放，从而控制family-wise的错误率。

> [如何通俗地理解Family-wise error rate(FWER)和False discovery rate(FDR)](https://blog.csdn.net/shengchaohua163/article/details/86738462)

### 相关系数

皮尔森相关系数是一种最简单的方法，能帮助理解两个连续变量之间的线性相关性。  

定义进度条


```python
class DropCorrelatedFeatures(Estimator, Transformer):
    def __init__(self, inputCols, threshold=0.9):
        self.inputCols = inputCols
        self.threshold = threshold

    @timer
    def _fit(self, df):
        inputCols = [col for col,dtype in df.dtypes if dtype not in ['string', 'vector']]
        to_keep = [inputCols[0]]
        to_drop = []
        for c1 in inputCols[1:]:
            # The correlations
            corr = df.select(*[fn.corr(c1, c2) for c2 in to_keep]).toPandas()
            # Select columns with correlations above threshold
            if np.any(corr.abs().gt(self.threshold)):
                to_drop.append(c1)
            else:
                to_keep.append(c1)
        self.to_drop = to_drop
        self.to_keep = to_keep
        return self
    
    def _transform(self, df):
        return df.drop(*self.to_drop)
```


```python
# Drops features that are correlated
# model = DropCorrelatedFeatures(features, threshold=0.9).fit(df)
# correlated = model.to_drop

# print(f'Dropped {len(correlated)} correlated features.')
```

上述函数速度较慢，最终选择使用spark自带的相关系数矩阵：


```python
from pyspark.ml.stat import Correlation

def drop_correlated_features(df, threshold=0.9):
    inputCols = [col for col,dtype in df.dtypes if dtype not in ['string', 'vector']]
    # Assemble the feature columns into a single vector column
    assembler = VectorAssembler(
        inputCols=inputCols,
        outputCol="numericFeatures"
    )
    df = assembler.transform(df)
    
    # Compute the correlation matrix with specified method using dataset.
    corrmat = Correlation.corr(df, 'numericFeatures', 'pearson').collect()[0][0]
    corrmat = pd.DataFrame(corrmat.toArray(), index=inputCols, columns=inputCols)
    
    # Upper triangle of correlations
    upper = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype('bool'))
    
    # Absolute value correlation
    corr = upper.unstack().dropna().abs()
    to_drop = corr[corr.gt(threshold)].reset_index()['level_1'].unique()
    
    return to_drop.tolist()
```


```python
correlated = drop_correlated_features(df.select(features))
selected_features = [col for col in features if col not in correlated]
print(f'Dropped {len(correlated)} correlated features.')       
```

    Dropped 127 correlated features.


### 卡方检验

卡方检验是一种用于衡量两个分类变量之间相关性的统计方法。


```python
# Find categorical features
int_features = [k for k,v in df.select(selected_features).dtypes if v == 'int']
vector_features = [k for k,v in df.select(selected_features).dtypes if v == 'vector']
nunique = df.select([fn.countDistinct(var).alias(var) for var in int_features]).first().asDict()

categorical_cols = [f for f, n in nunique.items() if n <= 50]
continuous_cols = list(set(selected_features) - set(categorical_cols + vector_features))
```


```python
from pyspark.ml.feature import UnivariateFeatureSelector

def chi2_test_selector(df, categoricalFeatures, outputCol):
    selector = UnivariateFeatureSelector(
        featuresCol="categoricalFeatures", 
        labelCol="label", 
        outputCol=outputCol,
        selectionMode="fdr"
    )
    selector.setFeatureType("categorical").setLabelType("categorical").setSelectionThreshold(0.05)
    
    # Assemble the feature columns into a single vector column
    assembler = VectorAssembler(
        inputCols=categoricalFeatures,
        outputCol="categoricalFeatures"
    )
    df = assembler.transform(df)
    model = selector.fit(df)
    df = model.transform(df)

    n = df.first()["categoricalFeatures"].size
    print("The number of dropped features:", n - len(model.selectedFeatures))
    return df

df_chi2_test = chi2_test_selector(df, categorical_cols + vector_features, 'selectedFeatures1')
```

    The number of dropped features: 32


### 方差分析

方差分析主要用于分类问题中连续特征的相关性。

如果针对分类问题，方差分析和卡方检验搭配使用，就能够完成一次完整的特征筛选，其中方差分析用于筛选连续特征，卡方检验用于筛选离散特征。


```python
def anova_selector(df, continuousFeatures, outputCol):
    selector = UnivariateFeatureSelector(
        featuresCol="continuousFeatures", 
        labelCol="label", 
        outputCol=outputCol,
        selectionMode="fdr"
    )
    selector.setFeatureType("continuous").setLabelType("categorical").setSelectionThreshold(0.05)
    
    # Assemble the feature columns into a single vector column
    assembler = VectorAssembler(
        inputCols=continuousFeatures,
        outputCol="continuousFeatures"
    )
    df = assembler.transform(df)
    model = selector.fit(df)
    df = model.transform(df)
    
    print("The number of dropped features:", len(continuousFeatures) - len(model.selectedFeatures))
    return df 

df_anova = anova_selector(df_chi2_test, continuous_cols, 'selectedFeatures2')
```

    The number of dropped features: 30


```python
_ = score_dataset(df_anova, inputCols=["selectedFeatures1", "selectedFeatures2"], nfold=2)
```

    Starting time	 21:49:20
    [0] train's areaUnderROC: 0.8526299026274513,  valid's areaUnderROC: 0.7632345170337489
    [1] train's areaUnderROC: 0.8533149455907856,  valid's areaUnderROC: 0.757047527015275
    cv_agg's valid auc: 0.7601 +/- 0.00309
    score_dataset cost 0 hours 4 minutes 16 seconds

```python
del df_chi2_test, df_anova
gc.collect()
```


    671

### 互信息

互信息是从信息熵的角度分析各个特征和目标之间的关系（包括线性和非线性关系）。


```python
@timer
def calc_mi_scores(df, inputCols, labelCol):
    mi_scores = pd.Series(name="MI Scores")
    n = df.count()
    y = labelCol
    for x in inputCols:
        grouped = df.groupBy(x, y).agg(fn.count("*").alias("Num_xy")).toPandas()
        grouped["Num_x"] = grouped.groupby(x)["Num_xy"].transform("sum")
        grouped["Num_y"] = grouped.groupby(y)["Num_xy"].transform("sum")
        grouped["MI"] = grouped["Num_xy"] / n * np.log(grouped["Num_xy"] / grouped["Num_x"] * n / grouped["Num_y"])
        grouped["MI"] = grouped["MI"].where(grouped["MI"] > 0, 0)
        
        mi_scores[x] = grouped["MI"].sum()
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores
```

上述代码中采用了离散变量的互信息计算方法，在此我们先将连续变量离散化。


```python
numBins = 50
buckets = {f"{col}_binned": col for col in continuous_cols}
bucketizer = ft.QuantileDiscretizer(
        numBuckets=numBins,
        handleInvalid='keep',
        inputCols=continuous_cols, 
        outputCols=list(buckets)
).fit(df)
df = bucketizer.transform(df)

discrete_cols = categorical_cols + list(buckets)
```


```python
class DropUninformative(Estimator, Transformer):
    def __init__(self, inputCols, labelCol="label", threshold=0.0):
        self.threshold = threshold
        self.inputCols = inputCols 
        self.labelCol = labelCol
    def _fit(self, df):
        mi_scores = calc_mi_scores(df, self.inputCols, self.labelCol)
        self.to_keep = mi_scores[mi_scores > self.threshold].index.tolist()
        self.to_drop = list(set(self.inputCols) - set(self.to_keep))
        return self
    def _transform(self, df):  
        return df.drop(*self.to_drop)
```


```python
model = DropUninformative(discrete_cols, "label", threshold=0.0).fit(df)
uninformative = [buckets.get(col, col) for col in model.to_drop]

print('The number of selected features:', len(model.to_keep))
print(f'Dropped {len(uninformative)} uninformative features.')
```


    Starting time	 21:54:53
    calc_mi_scores cost 0 hours 7 minutes 10 seconds
    The number of selected features: 229
    Dropped 5 uninformative features.        

### IV值

IV（Information Value）用来评价离散特征对二分类变量的预测能力。一般认为IV小于0.02的特征为无用特征。


```python
@timer
def calc_iv_scores(df, inputCols, labelCol="label"):
    assert df.select(labelCol).distinct().count() == 2, "y must be binary"
    iv_scores = pd.Series()
    
    # Compute information value
    for var in inputCols:
        grouped = df.groupBy(var).agg(
            fn.sum(labelCol).alias('Positive'),
            fn.count('*').alias('All')
        ).toPandas().set_index(var) 
        grouped['Negative'] = grouped['All']-grouped['Positive'] 
        grouped['Positive rate'] = grouped['Positive']/grouped['Positive'].sum()
        grouped['Negative rate'] = grouped['Negative']/grouped['Negative'].sum()
        grouped['woe'] = np.log(grouped['Positive rate']/grouped['Negative rate'])
        grouped['iv'] = (grouped['Positive rate']-grouped['Negative rate'])*grouped['woe']
        
        iv_scores[var] = grouped['iv'].sum()
    return iv_scores.sort_values(ascending=False)

iv_scores = calc_iv_scores(df, discrete_cols)
print(f"There are {iv_scores.le(0.02).sum()} features with iv <=0.02.")
```


    Starting time	 22:02:03
    calc_iv_scores cost 0 hours 6 minutes 38 seconds
    There are 98 features with iv <=0.02.

### 基尼系数

基尼系数用来衡量分类问题中特征对目标变量的影响程度。它的取值范围在0到1之间，值越大表示特征对目标变量的影响越大。常见的基尼系数阈值为0.02，如果基尼系数小于此阈值，则被认为是不重要的特征。


```python
@timer
def calc_gini_scores(df, inputCols, labelCol="label"):
    gini_scores = pd.Series()
    # Compute gini score
    for var in inputCols:
        p = df.groupBy(var).agg(
            fn.mean(labelCol).alias("mean")
            ).toPandas()
        gini = 1 - p['mean'].pow(2).sum()
        gini_scores[var] = gini    
    return gini_scores.sort_values(ascending=False)

gini_scores = calc_gini_scores(df, discrete_cols)
print(f"There are {gini_scores.le(0.02).sum()} features with gini <=0.02.")
```


    Starting time	 22:08:41
    calc_gini_scores cost 0 hours 7 minutes 41 seconds
    There are 1 features with gini <=0.02.

### VIF值

 VIF用于衡量特征之间的共线性程度。通常，VIF小于5被认为不存在多重共线性问题，VIF大于10则存在明显的多重共线性问题。


```python
def calc_vif_scores(df):
    pass

# vif_scores = calc_vif_scores(df)
# print(f"There are {vif_scores.gt(10).sum()} collinear features (VIF above 10)")
```

### 小结

最终，我们选择删除高相关特征和无信息特征。


```python
features_to_drop = list(set(uninformative) | set(correlated))
selected_features = [col for col in features if col not in features_to_drop]

print('The number of selected features:', len(selected_features))
print(f'Dropped {len(features_to_drop)} features.')
```

    The number of selected features: 239
    Dropped 132 features.


在371个总特征中只保留了239个，表明我们创建的许多特征是多余的。

## 递归消除特征

最常用的包装法是递归消除特征法(recursive feature elimination)。递归消除特征法使用一个机器学习模型来进行多轮训练，每轮训练后，消除最不重要的特征，再基于新的特征集进行下一轮训练。

由于RFE需要消耗大量的资源，这里就不编写函数运行了。

## 特征重要性

嵌入法也是用模型来选择特征，但是它和RFE的区别是它不通过不停的筛掉特征来进行训练，而是使用特征全集训练模型。

- 最常用的是使用带惩罚项（$\ell_1,\ell_2$ 正则项）的基模型，来选择特征，例如 Lasso，Ridge。
- 或者简单的训练基模型，选择权重较高的特征。

我们先使用之前定义的 `score_dataset` 获取每个特征的重要性分数：


```python
feature_importances = score_dataset(df, inputCols=selected_features, nfold=2)
```

    Starting time	 22:16:22
    [0] train's areaUnderROC: 0.8545613660810463,  valid's areaUnderROC: 0.7633448519087491
    [1] train's areaUnderROC: 0.8553078656308732,  valid's areaUnderROC: 0.7570120756115536
    cv_agg's valid auc: 0.7602 +/- 0.00317
    score_dataset cost 0 hours 4 minutes 24 seconds


```python
# Sort features according to importance
feature_importances = feature_importances.sort_values('fscore', ascending=False)
feature_importances['fscore'].head(15)
```


    AMT_GOODS_PRICE/AMT_ANNUITY       1448.0
    DEF_60_CNT_SOCIAL_CIRCLE          1061.0
    AMT_GOODS_PRICE/AMT_CREDIT         924.0
    ln(EXT_SOURCE_2)                   867.0
    ln(EXT_SOURCE_3)                   793.5
    ORGANIZATION_TYPE/DAYS_BIRTH       777.5
    DAYS_BIRTH/EXT_SOURCE_1            776.5
    EXT_SOURCE_3/ORGANIZATION_TYPE     723.0
    EXT_SOURCE_3/DAYS_BIRTH            687.5
    ORGANIZATION_TYPE/EXT_SOURCE_1     685.5
    centroid_0                         662.5
    EXT_SOURCE_2/ORGANIZATION_TYPE     658.5
    EXT_SOURCE_2/DAYS_BIRTH            635.0
    AMT_ANNUITY/AMT_INCOME_TOTAL       622.5
    EXT_SOURCE_1/DAYS_BIRTH            587.0
    Name: fscore, dtype: float64

可以看到，我们构建的许多特征进入了前15名，这应该让我们有信心，我们所有的辛勤工作都是值得的！

接下来，我们删除重要性为0的特征，因为这些特征实际上从未用于在任何决策树中拆分节点。因此，删除这些特征是一个非常安全的选择（至少对这个特定模型来说）。


```python
# Find the features with zero importance
zero_importance = feature_importances.query("fscore == 0.0").index.tolist()
print(f'\nThere are {len(zero_importance)} features with 0.0 importance')
```


    There are 7 features with 0.0 importance

```python
selected_features = [col for col in selected_features if col not in zero_importance]
print("The number of selected features:", len(selected_features))
print("Dropped {} features with zero importance.".format(len(zero_importance)))
```

    The number of selected features: 232
    Dropped 7 features with zero importance.

删除0重要性的特征后，我们还有232个特征。如果我们认为此时特征量依然非常大，我们可以继续删除重要性最小的特征。   
下图显示了累积重要性与特征数量：


```python
feature_importances = feature_importances.sort_values('fscore', ascending=False)

sns.lineplot(x=range(1, feature_importances.shape[0]+1), y=feature_importances['fscore'].cumsum())
plt.show()
```


![](/img/feature_engineering_on_spark/selection_output_47_0.png)


如果我们选择是只保留95%的重要性所需的特征：


```python
def select_import_features(scores, thresh=0.95):
    feature_imp = pd.DataFrame({'score': feature_importances['fscore']})
    # Sort features according to importance
    feature_imp = feature_imp.sort_values('score', ascending=False)
    # Normalize the feature importances
    feature_imp['score_normalized'] = feature_imp['score'] / feature_imp['score'].sum()
    feature_imp['cumsum'] = feature_imp['score_normalized'].cumsum()
    selected_features = feature_imp.query(f'cumsum <= {thresh}')
    return selected_features.index.tolist()

import_features = select_import_features(feature_importances['fscore'], thresh=0.95)
print("The number of import features:", len(import_features))
print(f'Dropped {len(selected_features) - len(import_features)} features.')
```

    The number of import features: 157
    Dropped 75 features.


剩余157个特征足以覆盖95%的重要性。


```python
feature_importances = score_dataset(df, inputCols=import_features)
```

    Starting time	 22:20:46
    [0] train's areaUnderROC: 0.8645996680043095,  valid's areaUnderROC: 0.7537419087196509
    [1] train's areaUnderROC: 0.8617688316741262,  valid's areaUnderROC: 0.7494887919280331
    [2] train's areaUnderROC: 0.8602027702822611,  valid's areaUnderROC: 0.7489103752356694
    cv_agg's valid auc: 0.7507 +/- 0.00215
    score_dataset cost 0 hours 3 minutes 58 seconds


 在继续之前，我们应该记录我们采取的特征选择步骤，以备将来使用：

1. 删除互信息为0的无效特征：删除了5个特征
2. 删除相关系数大于0.9的共线变量：删除了127个特征
3. 根据GBM删除0.0重要特征：删除7个特征
4. (可选)仅保留95%特征重要性所需的特征：删除了75个特征

我们看下特征组成：


```python
original_df = spark.sql("select * from home_credit_default_risk.prepared_data").limit(1).toPandas()

original_features = [f for f in selected_features if f in original_df.columns]
derived_features =  [f for f in selected_features if f not in original_features]

print(f"Selected features: {len(original)} original features, {len(derived)} derived features.")
```

    Selected features: 79 original features, 153 derived features.


保留的222个特征，有79个是原始特征，153个是衍生特征。

## 主成分分析

常见的降维方法除了基于L1惩罚项的模型以外，另外还有主成分分析法（PCA）和线性判别分析（LDA）。这两种方法的本质是相似的，本节主要介绍PCA。


```python
pca = ft.PCA(
    k=len(features), 
    inputCol="scaled", 
    outputCol="pcaFeatures"
)
# Assemble the feature columns into a single vector column
assembler = VectorAssembler(
    inputCols=features,
    outputCol="features"
)
scaler = ft.RobustScaler(
    inputCol="features",
    outputCol="scaled"
)

pipeline = Pipeline(stages=[assembler, scaler, pca]).fit(df)
```


```python
pcaModel = pipeline.stages[2]
print("explained variance ratio:\n", pcaModel.explainedVariance[:5])

pca_df = pipeline.transform(df)
weight_matrix = pcaModel.pc
```

    explained variance ratio:
     [9.47148918e-01 4.88162534e-02 3.38563499e-03 2.82225779e-04
     7.15668020e-05]


其中 `pcaModel.pc` 对应 PCA 求解矩阵的SVD分解的截断矩阵 $V$，形状为 `(n_features, n_components)` ，其中 `n_components` 是我们指定的主成分数目，`n_features` 是原始数据的特征数目。`pcaModel.pc` 的每一列表示一个主成分，每一行表示原始数据的一个特征。因此，`pca.components_` 的每个元素表示对应特征在主成分中的权重。

可视化方差


```python
def plot_variance(pca, n_components=10):
    evr = pca.explainedVariance[:n_components]
    grid = range(1, n_components + 1)
    
    # Create figure
    plt.figure(figsize=(6, 4))
       
    # Percentage of variance explained for each components.
    plt.bar(grid, evr, label='Explained Variance')
    
    # Cumulative Variance
    plt.plot(grid, np.cumsum(evr), "o-", label='Cumulative Variance', color='orange')  
    plt.xlabel("The number of Components")
    plt.xticks(grid)
    plt.title("Explained Variance Ratio")
    plt.ylim(0.0, 1.1)
    plt.legend(loc='best')

plot_variance(pcaModel)
plt.show()
```

![](/img/feature_engineering_on_spark/selection_output_59_0.png)
  PCA可以有效地减少维度的数量，但他们的本质是要将原始的样本映射到维度更低的样本空间中。这意味着PCA特征没有真正的业务含义。此外，PCA假设数据是正态分布的，这可能不是真实数据的有效假设。因此，我们只是展示了如何使用pca，实际上并没有将其应用于数据。

## 总结

本章介绍了很多特征选择方法
1. 单变量特征选择可以用于理解数据、数据的结构、特点，也可以用于排除不相关特征，但是它不能发现冗余特征。
2. 正则化的线性模型可用于特征理解和特征选择。但是它需要先把特征转换成正态分布。
3. 嵌入法的特征重要性选择是一种非常流行的特征选择方法，它易于使用。但它有两个主要问题：
   - 重要的特征有可能得分很低（关联特征问题）
   - 这种方法对类别多的特征越有利（偏向问题）

至此，经典的特征工程至此已经完结了，我们继续使用XGBoost模型评估筛选后的特征。


```python
feature_importances = score_dataset(df, selected_features, nfold=2)
```

    Starting time	 22:27:24
    [0] train's areaUnderROC: 0.8514944460937176,  valid's areaUnderROC: 0.7609365503074478
    [1] train's areaUnderROC: 0.8528487869720561,  valid's areaUnderROC: 0.7552742165606054
    cv_agg's valid auc: 0.7581 +/- 0.00283
    score_dataset cost 0 hours 4 minutes 8 seconds


特征重要性：


```python
# Sort features according to importance
feature_importances['fscore'].sort_values(ascending=False).head(15)
```


    AMT_GOODS_PRICE/AMT_ANNUITY       1351.0
    ln(EXT_SOURCE_2)                   878.0
    AMT_GOODS_PRICE/AMT_CREDIT         876.5
    ORGANIZATION_TYPE/DAYS_BIRTH       840.0
    EXT_SOURCE_3/DAYS_BIRTH            749.0
    DAYS_BIRTH/EXT_SOURCE_1            724.5
    centroid_0                         689.0
    ln(EXT_SOURCE_3)                   681.0
    EXT_SOURCE_2/DAYS_BIRTH            675.5
    EXT_SOURCE_1/DAYS_BIRTH            670.0
    EXT_SOURCE_3/ORGANIZATION_TYPE     659.5
    AMT_REQ_CREDIT_BUREAU_QRT          638.0
    EXT_SOURCE_2/ORGANIZATION_TYPE     635.5
    AMT_ANNUITY/AMT_INCOME_TOTAL       629.5
    ORGANIZATION_TYPE/EXT_SOURCE_1     593.5
    Name: fscore, dtype: float64

保存数据集


```python
selected_data = df.select('SK_ID_CURR', 'label', *selected_features)
selected_data.write.bucketBy(100, "SK_ID_CURR").mode("overwrite").saveAsTable("home_credit_default_risk.selected_data")                                 
```


```python
spark.stop()
```
