---
title: PySpark 特征工程(II)--特征构造
tags:
  - Python
  - Spark
categories:
  - Big Data
  - Spark
cover: /img/apache-spark-mllib.png
top_img: /img/apache-spark-top-img.svg
abbrlink: a1358f89
description: 
date: 2024-06-03 21:45:02
---

有这么一句话在业界广泛流传：数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。由此可见，特征工程在机器学习中占有相当重要的地位。在实际应用当中，可以说特征工程是机器学习成功的关键。

特征工程是数据分析中最耗时间和精力的一部分工作，它不像算法和模型那样是确定的步骤，更多是工程上的经验和权衡。因此没有统一的方法。这里只是对一些常用的方法做一个总结。

特征工程包含了 Data PreProcessing（数据预处理）、Feature Extraction（特征提取）、Feature Selection（特征选择）和 Feature construction（特征构造）等子问题。

Jupyter Notebook 代码连接：[feature_engineering_on_spark_p2_feature_construction](/ipynb/feature_engineering_on_spark_p2_feature_construction)

# 特征构造

特征构造是从现有数据创建新特征的过程。目标是构建有用的功能，帮助我们的模型了解数据集中的信息与给定目标之间的关系。


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

    24/06/03 20:13:12 WARN Utils: Your hostname, MacBook-Air resolves to a loopback address: 127.0.0.1; using 192.168.1.5 instead (on interface en0)
    24/06/03 20:13:12 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    24/06/03 20:13:13 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable

```python
df = spark.sql("select * from home_credit_default_risk.prepared_data")
```

    Loading class `com.mysql.jdbc.Driver'. This is deprecated. The new driver class is `com.mysql.cj.jdbc.Driver'. The driver is automatically registered via the SPI and manual loading of the driver class is generally unnecessary.

```python
# Number of each type of column
dtypes = dict(df.dtypes)
pd.Series(dtypes).value_counts()
```


    double    65
    int       45
    vector    10
    float      2
    Name: count, dtype: int64

为了检测新特征集的性能，我们先定义数据集评估函数。


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

## 简单数学变换

我们可以根据业务含义，创建具有一些明显实际含义的补充特征，例如：


```python
math_features = df

# 贷款金额相对于收入的比率
math_features = math_features.withColumn('CREDIT_INCOME_PERCENT', df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'])

# 贷款年金占总收入比率
math_features = math_features.withColumn('ANNUITY_INCOME_PERCENT', df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'])

# 以月为单位的付款期限
math_features = math_features.withColumn('CREDIT_TERM', df['AMT_ANNUITY'] / df['AMT_CREDIT']) 

#工作时间占年龄的比率
math_features = math_features.withColumn('DAYS_EMPLOYED_PERCENT', df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']) 

# 该用户家庭的人均收入
math_features = math_features.withColumn('INCOME_PER_PERSON', df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']) 
```

我们可以在图形中直观地探索这些新变量：


```python
new_cols = ['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']
math_features = math_features.select(*new_cols, 'label').sample(0.1).toPandas()

plt.figure(figsize = (10, 6))
# iterate through the new features
for i, feature in enumerate(new_cols):
    # create a new subplot for each feature
    plt.subplot(2, 2, i + 1)
    sns.kdeplot(data=math_features, x=feature, hue='label', common_norm=False)   
```

![](/img/feature_engineering_on_spark/construction_output_12_1.png)

```python
del math_features
gc.collect()
```

当然，我们不可能手动计算出所有有实际含义的数学特征。我们编写一个函数找到尽可能多的特征组合进行加减乘除运算。另外，还有对单变量的对数、指数、倒数、平方根、三角函数等运算。


```python
from itertools import product, permutations, combinations

class mathFeatureCreator(Estimator, Transformer):
    """
    Apply math operators to create new features.
    Parameters
    ----------
    inputCols: List[string]
        The list of input variables.
    operators: List[string]
        List of Transform Feature functions to apply.
        - binary: 'add', 'subtract', 'multiply', 'divide'
        - unary: 'sin', 'cos', 'tan', 'sqrt', 'ln'   
    """
    def __init__(self, inputCols, operators):    
        self.inputCols = inputCols
        self.operators = operators    
    
    def _fit(self, df):
        return self
    
    def _transform(self, df):
        funcs = {
            'add': (fn.try_add, '+'),
            'subtract': (fn.try_subtract, '-'),
            'multiply': (fn.try_multiply, '*'),
            'divide': (fn.try_divide, '/'),
            'sin': fn.sin,
            'cos': fn.cos, 
            'tan': fn.tan, 
            'sqrt': fn.sqrt, 
            'ln': fn.ln
        }
        
        commutation = [o for o in self.operators if o in ['add', 'subtract', 'multiply']]
        binary = [o for o in self.operators if o in ['divide']]
        unary = [o for o in self.operators if o in ['sin', 'cos', 'tan', 'sqrt', 'ln']]
        
        comb = list(combinations(self.inputCols, 2))
        perm = list(permutations(self.inputCols, 2))
        
        feature_number = (len(list(comb)) * len(commutation) + 
              len(list(perm)) * len(binary) + 
              len(unary) * len(self.inputCols))
        print(f"Built {feature_number} features.")
        
        df_new = df.select("SK_ID_CURR", *self.inputCols)
        for operator, (left, right) in product(commutation, comb):
            func = funcs[operator][0]
            colname = left + funcs[operator][1] + right
            df_new = df_new.withColumn(colname, func(left, right))
        
        for operator, (left, right) in product(binary, perm):
            func = funcs[operator][0]
            colname = left + funcs[operator][1] + right
            df_new = df_new.withColumn(colname, func(left, right))
        
        for operator, col in product(unary, self.inputCols):
            func = funcs[operator]
            colname = f"{operator}({col})"
            df_new = df_new.withColumn(colname, func(col))
        
        return df_new.drop(*self.inputCols)
```

注意：对于二元运算虽然简单，但会出现阶乘式维度爆炸。因此，我们挑选出少数型特征进行简单运算。


```python
features_to_trans = [
    'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'AMT_GOODS_PRICE', 
    'DAYS_ID_PUBLISH', 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE',
    'DAYS_REGISTRATION', 'AMT_INCOME_TOTAL', 'REGION_POPULATION_RELATIVE', 
    'ORGANIZATION_TYPE', 'OWN_CAR_AGE', 'OCCUPATION_TYPE', 
    'HOUR_APPR_PROCESS_START', 'NAME_EDUCATION_TYPE', 'NONLIVINGAPARTMENTS_MEDI',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
]
```


```python
nunique = df.select([fn.countDistinct(col).alias(col) for col in features_to_trans]).first()

discrete_to_trans = [f for f in features_to_trans if nunique[f] < 50]
continuous_to_trans = [f for f in features_to_trans if nunique[f] >= 50]

print("Selected discrete features:", discrete_to_trans, sep='\n')
print("Selected continuous features:", continuous_to_trans, sep='\n')
```


    Selected discrete features:
    ['OCCUPATION_TYPE', 'HOUR_APPR_PROCESS_START', 'NAME_EDUCATION_TYPE']
    Selected continuous features:
    ['AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'AMT_GOODS_PRICE', 'DAYS_ID_PUBLISH', 'DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_REGISTRATION', 'AMT_INCOME_TOTAL', 'REGION_POPULATION_RELATIVE', 'ORGANIZATION_TYPE', 'OWN_CAR_AGE', 'NONLIVINGAPARTMENTS_MEDI', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']            


```python
math_feature_creator = mathFeatureCreator(
    inputCols=continuous_to_trans, 
    operators=['divide', 'sqrt', 'ln']
)
math_features = math_feature_creator.transform(df)
```

    Built 272 features.


## 分组统计特征衍生

分组统计特征衍生，顾名思义，就是分类特征和连续特征间的分组交互统计，这样可以得到更多有意义的特征，例如：


```python
# Group AMT_INCOME_TOTAL by NAME_INCOME_TYPE and calculate mean, max, min of loans
df.groupBy('OCCUPATION_TYPE').agg(
    fn.mean('AMT_INCOME_TOTAL').alias('mean'), 
    fn.max('AMT_INCOME_TOTAL').alias('max'), 
    fn.min('AMT_INCOME_TOTAL').alias('min')
).show(5)
```

    +---------------+------------------+---------+-------+
    |OCCUPATION_TYPE|              mean|      max|    min|
    +---------------+------------------+---------+-------+
    |     0.07856473|195003.99467376832| 675000.0|67500.0|
    |     0.06396721|188916.28241563056| 699750.0|30600.0|
    |     0.06599255|182334.81278280544|3150000.0|36000.0|
    |     0.10578712|166357.48252518754|   1.17E8|27000.0|
    |    0.061600607|182842.04568321616|1890000.0|27000.0|
    +---------------+------------------+---------+-------+
    only showing top 5 rows

常用的统计量
|||
|:--|:---|
|var/std|方差、标准差|
|mean/median|均值、中位数|
|max/min|最大值、最小值|
|skew|偏度|
|mode|众数|
|nunique|类别数|
|frequency|频数|
|count|个数|
|quantile|分位数|

> 注意：分组特征必须是离散特征，且最好是一些取值较多的离散变量，这样可以避免新特征出现大量重复取值。分组使用连续值特征时一般需要先进行离散化。

接下来我们自定义一个transformer用来处理数值类型和分类型的分组变量衍生。

定义计时器和进度条


```python
def progress(percent=0, width=50, desc="Processing"):
    import math
    tags = math.ceil(width * percent) * "#"
    print(f"\r{desc}: [{tags:-<{width}}]{percent:.1%}", end="", flush=True)
```


```python
class AggFeatures(Estimator, Transformer):
    """
    Transformer to aggregate features in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    ----------
    inputCols: List[string]
        The list of input variables. At least one of `variables`, 
    groupBy: List[string]
        The variables to group by. 
    funcs: List[string]
        List of Aggregation Feature types to apply. 
        - Numeric: ['mean', 'median', 'max', 'min', 'skew', 'std', 'kurt']
        - Category: ['mode', 'num_unique']
    numBins: int, default=10
        The number of bins to produce.
    """
    
    def __init__(self, inputCols, funcs, groupByDiscrete=None, groupByContinuous=None, numBins=10):  
        self.inputCols = inputCols
        self.funcs = funcs
        self.groupByDiscrete = groupByDiscrete
        self.groupByContinuous = groupByContinuous
        self.numBins= numBins
    
    def _fit(self, df): 
        return self 

    @timer
    def _transform(self, df):
        groupBy = []
        if self.groupByContinuous is not None:
            outputCols = [f"{col}_binned" for col in self.groupByContinuous]
            bucketizer = ft.QuantileDiscretizer(
                numBuckets=self.numBins,
                handleInvalid='keep',
                inputCols=self.groupByContinuous, 
                outputCols=outputCols
            ).fit(df)
            df = bucketizer.transform(df)
            groupBy.extend(outputCols)
        if self.groupByDiscrete is not None:
            groupBy.extend(self.groupByDiscrete)
        if len(groupBy) == 0:
            raise ValueError("groupBy is None.")
         
        # Group by the specified variable and calculate the statistics
        mapping = {
            'mean': fn.mean, 
            'median': fn.median,
            'max': fn.max,
            'min': fn.min,
            'skew': fn.skewness,
            'kurt': fn.kurtosis,
            'std': fn.std,
            'mode': fn.mode, 
            'num_unique': fn.countDistinct
        }
        new_cols = []
        i = 0
        for by in groupBy:
            # Skip the grouping variable
            other_vars = [var for var in self.inputCols if by not in [var, f"{var}_binned"]]
            for f in self.funcs:
                colnames = {col: f"{f}({col})_by({by})" for col in other_vars}
                new_cols.extend(colnames.values())
                
                f = mapping[f]
                grouped = df.groupBy(by).agg(*[f(df[var]).alias(name) for var, name in colnames.items()])
                df = df.join(grouped.dropna(), on=by, how='left')
                
                i += 1
                progress(i / len(groupBy) / len(self.funcs))
        print(f"Created {len(new_cols)} new features.")     
        return df.select("SK_ID_CURR", *new_cols)
```

数值型特征直接计算统计量，至于分类特征，计算众数、类别数。


```python
agg_continuous_transformer = AggFeatures(
    inputCols=continuous_to_trans,
    funcs=['mean'],
    groupByDiscrete=discrete_to_trans, 
    groupByContinuous=continuous_to_trans
)
agg_discrete_transformer = AggFeatures(
    inputCols=discrete_to_trans,
    funcs=['mode'],
    groupByDiscrete=discrete_to_trans, 
    groupByContinuous=continuous_to_trans
)
agg_transformer = Pipeline(stages=[agg_continuous_transformer, agg_discrete_transformer])
# agg_features = agg_transformer.fit(df).transform(df)
# print(agg_features.columns)
```

## 特征交互

通过将单独的特征求笛卡尔乘积的方式来组合2个或更多个特征，从而构造出组合特征。最终获得的预测能力可能远远超过任一特征单独的预测能力。笛卡尔乘积组合特征方法一般应用于离散特征之间。


```python
@timer
def feature_interaction(df, left, right):
    """
    Parameters
    ----------
    df: pyspark dataframe.
    left, right: The list of interact variables. default=None
    """
    print(f"Built {len(left) * len(right)} features.")
    # Make a new dataframe to hold interaction features
    inputCols = set(left + right)
    df_new = df.select("SK_ID_CURR", *inputCols)
    i = 0
    for rvar in right:
        df_new = df_new.withColumns({f"{lvar}&{rvar}": fn.concat_ws('&', df[lvar], df[rvar]) for lvar in left if lvar !=rvar})
        i += 1
        progress(i / len(right))
    return df_new.drop(*inputCols)

feature_interaction(df.limit(5), discrete_to_trans, discrete_to_trans).toPandas()
```

    Starting time	 20:13:41
    Built 9 features.
    Processing: [##################################################]100.0%feature_interaction cost 0 hours 0 minutes 0 seconds

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SK_ID_CURR</th>
      <th>HOUR_APPR_PROCESS_START&amp;OCCUPATION_TYPE</th>
      <th>NAME_EDUCATION_TYPE&amp;OCCUPATION_TYPE</th>
      <th>OCCUPATION_TYPE&amp;HOUR_APPR_PROCESS_START</th>
      <th>NAME_EDUCATION_TYPE&amp;HOUR_APPR_PROCESS_START</th>
      <th>OCCUPATION_TYPE&amp;NAME_EDUCATION_TYPE</th>
      <th>HOUR_APPR_PROCESS_START&amp;NAME_EDUCATION_TYPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100180</td>
      <td>15&amp;0.048305318</td>
      <td>3&amp;0.048305318</td>
      <td>0.048305318&amp;15</td>
      <td>3&amp;15</td>
      <td>0.048305318&amp;3</td>
      <td>15&amp;3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100201</td>
      <td>10&amp;0.06304005</td>
      <td>3&amp;0.06304005</td>
      <td>0.06304005&amp;10</td>
      <td>3&amp;10</td>
      <td>0.06304005&amp;3</td>
      <td>10&amp;3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100243</td>
      <td>13&amp;0.10578712</td>
      <td>3&amp;0.10578712</td>
      <td>0.10578712&amp;13</td>
      <td>3&amp;13</td>
      <td>0.10578712&amp;3</td>
      <td>13&amp;3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100435</td>
      <td>10&amp;0.09631742</td>
      <td>1&amp;0.09631742</td>
      <td>0.09631742&amp;10</td>
      <td>1&amp;10</td>
      <td>0.09631742&amp;1</td>
      <td>10&amp;1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100441</td>
      <td>16&amp;0.09631742</td>
      <td>1&amp;0.09631742</td>
      <td>0.09631742&amp;16</td>
      <td>1&amp;16</td>
      <td>0.09631742&amp;1</td>
      <td>16&amp;1</td>
    </tr>
  </tbody>
</table>
</div>

## 多项式特征

多项式特征是 sklearn 中特征构造的最简单方法。当我们创建多项式特征时，尽量避免使用过高的度数，因为特征的数量随着度数指数级地变化，并且可能过拟合。

现在我们使用3度多项式来查看结果：


```python
from pyspark.ml.feature import PolynomialExpansion

# Make a new dataframe for polynomial features
assembler = VectorAssembler(
    inputCols=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'],
    outputCol="ext_features"
)
poly_df = assembler.transform(df)

# Create the polynomial object with specified degree
poly_transformer = PolynomialExpansion(
    degree=3,
    inputCol='ext_features',
    outputCol='poly_expanded'
)

# Train and transform the polynomial features
poly_transformer.transform(poly_df).select('poly_expanded').show(5)
```

    +--------------------+
    |       poly_expanded|
    +--------------------+
    |[0.72908990917097...|
    |[0.76491718757837...|
    |[0.50549856415141...|
    |[0.50549856415141...|
    |[0.50549856415141...|
    +--------------------+
    only showing top 5 rows


```python
del poly_df
gc.collect()
```


    13801

## 聚类分析

聚类算法在特征构造中的应用有不少，例如：利用聚类算法对文本聚类，使用聚类类标结果作为输入特征；利用聚类算法对单个数值特征进行聚类，相当于分箱；利用聚类算法对R、F、M数据进行聚类，类似RFM模型，然后再使用代表衡量客户价值的聚类类标结果作为输入特征。

当一个或多个特征具有多峰分布（有两个或两个以上清晰的峰值）时，可以使用聚类算法为每个峰值分类，并输出聚类类标结果。


```python
age_simi = df.select((df['DAYS_BIRTH']/-365).alias('age'))
plt.figure(figsize=(5,4))
sns.histplot(x=age_simi.sample(0.1).toPandas()['age'], bins=30)
```


![](/img/feature_engineering_on_spark/construction_output_33_1.png)


可以看到有两个峰值：40和55

一般聚类类标结果为一个数值，但实际上这个数值并没有大小之分，所以一般需要进行one-hot编码，或者创建新特征来度量样本和每个类中心的相似性（距离）。相似性度量通常使用径向基函数(RBF)来计算。

径向基函数（Radial Basis Function，简称RBF）是一种在机器学习和统计建模中常用的函数类型。它们以距离某个中心点的距离作为输入，并输出一个数值，通常表示为距离的“衰减函数”。最常用的RBF是高斯RBF，其输出值随着输入值远离固定点而呈指数衰减。接下来，我们先定义高斯RBF函数 $k(x,y)=\exp(-\gamma\|x-y\|^2)$，超参数 gamma 确定当x远离y时相似性度量衰减的速度。


```python
from pyspark.ml.linalg import Vector
def rbf_kernel(row, vector1, vectors2, outputCols, gamma=1.0):
    if isinstance(vector1, str):
        v1 = row[vector1]
    else:
        v1 = Vectors.dense([row[i] for i in vector1])
    
    row = row.asDict()
    for i, v2 in enumerate(vectors2):
        v2 = Vectors.dense(v2)
        squared_distance = v1.squared_distance(v2)
        rbf = np.exp(-gamma * squared_distance)
        row[outputCols[i]] = float(rbf)
    return Row(**row)
```

下图显示了年龄的两个径向基函数：


```python
age_simi = age_simi.rdd.map(lambda x: rbf_kernel(x, ['age'], [[40], [55]], ['simi_40', 'simi_55'], gamma=0.01)).toDF()

age_simi_pdf = age_simi.sample(0.1).toPandas()
fig = plt.figure(figsize=(5,4))
ax1 = fig.add_subplot(111)
sns.histplot(data=age_simi_pdf, x='age', bins=30, ax=ax1)

ax2 = ax1.twinx()
sns.lineplot(data=age_simi_pdf, x='age', y='simi_40', ci=None, ax=ax2, color='green')
sns.lineplot(data=age_simi_pdf, x='age', y='simi_55', ci=None, ax=ax2, color='orange')
```

![](/img/feature_engineering_on_spark/construction_output_37_2.png)

```python
del age_simi, age_simi_pdf
gc.collect()
```


    151

如果这个特定的特征与目标变量有很好的相关性，那么这个新特征将有很好的机会发挥作用。

接下来，我们自定义一个函数使用`rbf_kernel()` 来衡量每个样本与每个聚类中心的相似程度：


```python
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import RobustScaler

def ClusterSimilarity(df, inputCols, outputCols, n_clusters=5, gamma=1.0, seed=None):
    # Assemble the feature columns into a single vector column
    assembler = VectorAssembler(
        inputCols=inputCols, 
        outputCol="features"
    )
    # Standardize
    scaler = RobustScaler(inputCol="features", outputCol="scaled")
    # Trains a k-means model.
    kmeans = KMeans(featuresCol="scaled", k=n_clusters, seed=seed)
    pipeline = Pipeline(stages=[assembler, scaler, kmeans]).fit(df)
    kmeans = pipeline.stages[2]
    # Coordinates of cluster centers.
    clusterCenters = [list(i) for i in kmeans.clusterCenters()]
     
    df = pipeline.transform(df).select("SK_ID_CURR", "scaled")
    df = df.rdd.map(lambda x: rbf_kernel(x, "scaled", clusterCenters, outputCols, gamma=gamma)).toDF()
    return df.drop("scaled")
```

现在让我们使用这个转换器：


```python
cluster_features = ClusterSimilarity(
    df=df,
    inputCols=df.drop("SK_ID_CURR",'label').columns, 
    outputCols=[f"centroid_{i}" for i in range(5)], 
    n_clusters=5, 
    gamma=1.0, 
    seed=SEED
)

cluster_features.show(5)                   
```

    +----------+--------------------+----------+----------+----------+----------+
    |SK_ID_CURR|          centroid_0|centroid_1|centroid_2|centroid_3|centroid_4|
    +----------+--------------------+----------+----------+----------+----------+
    |    100180|3.712547879890347...|       0.0|       0.0|       0.0|       0.0|
    |    100201|3.217366500721083...|       0.0|       0.0|       0.0|       0.0|
    |    100243|1.598249005834992...|       0.0|       0.0|       0.0|       0.0|
    |    100435|3.543010683726346...|       0.0|       0.0|       0.0|       0.0|
    |    100441|1.021260362527048...|       0.0|       0.0|       0.0|       0.0|
    +----------+--------------------+----------+----------+----------+----------+
    only showing top 5 rows

## 主成分分析

由于我们新增的这些特征都是和原始特征高度相关，可以使用PCA的主成分作为新的特征来消除相关性。

由于主成分分析主要用于降维，我们将在后续特征选择部分详细介绍。

## 总结

合并之前创造的特征


```python
# Combine datasets
df_created = (df.join(math_features, on='SK_ID_CURR') 
                # .join(agg_features, on='SK_ID_CURR')
                .join(cluster_features, on='SK_ID_CURR')
             )
```

缺失值处理


```python
# Remove variables with high missing rate
def drop_missing_data(df, threshold=0.8):
    # Remove variables with missing more than threshold(default 20%)
    thresh = int(df.count() * (1 - threshold))
    exprs = [fn.sum(df[col].isNull().cast('int')).alias(col) for col in df.columns]
    missing_number = df.select(*exprs).first().asDict()
    cols_to_drop = [k for k,v in missing_number.items() if v > thresh]
    print(f"Removed {len(cols_to_drop)} variables with missing more than {1 - threshold:.1%}")
    return df.drop(*cols_to_drop)

def handle_missing(df):
    # Remove variables with high missing rate
    df = drop_missing_data(df, threshold=0.8)
    
    # Univariate imputer for completing missing values with simple strategies.
    dtypes = df.drop("SK_ID_CURR", "label").dtypes
    numerical_cols = [k for k, v in dtypes if v not in ('string', 'vector')]
    imputed_cols = [f"imputed_{col}" for col in numerical_cols]
    imputer = ft.Imputer(
        inputCols=numerical_cols,
        outputCols=imputed_cols,
        strategy="median"
    )
    df = imputer.fit(df).transform(df)
    colsMap = dict(zip(imputed_cols, numerical_cols))
    df = df.drop(*numerical_cols).withColumnsRenamed(colsMap)
    return df
```


```python
df_created = handle_missing(df_created.replace(np.nan, None))
missing_num = df_created.select([df_created[var].isNull().cast('int') for var in df_created.columns])
missing_num = pd.Series(missing_num.first().asDict())
print(f"The dataframe has {missing_num.sum()} columns that have missing values.")
```

    Removed 26 variables with missing more than 20.0%
    The dataframe has 0 columns that have missing values.


```python
print(f"dataset shape: ({df_created.count()}, {len(df_created.columns)})")          
```

    dataset shape: (307511, 373)


我们继续使用XGBoost模型评估创造的新特征


```python
features = df_created.drop('SK_ID_CURR', 'label').columns
feature_importances = score_dataset(df_created, inputCols=features, nfold=2)
```


    Starting time	 20:18:16
    [0] train's areaUnderROC: 0.8589685970790397,  valid's areaUnderROC: 0.7649915434701238
    [1] train's areaUnderROC: 0.8599883799618516,  valid's areaUnderROC: 0.7598499769233037
    cv_agg's valid auc: 0.7624 +/- 0.00257
    score_dataset cost 0 hours 12 minutes 29 seconds


特征重要性


```python
feature_importances['fscore'].head(20)
```


    DAYS_ID_PUBLISH/AMT_CREDIT            981.0
    AMT_GOODS_PRICE/AMT_CREDIT            980.0
    DAYS_ID_PUBLISH/DAYS_BIRTH            680.5
    YEARS_BEGINEXPLUATATION_MODE          581.0
    DAYS_EMPLOYED/EXT_SOURCE_1            555.0
    NONLIVINGAPARTMENTS_MEDI              542.0
    DAYS_LAST_PHONE_CHANGE/AMT_ANNUITY    530.0
    DAYS_LAST_PHONE_CHANGE/AMT_CREDIT     528.0
    NONLIVINGAREA_MEDI                    509.0
    DAYS_EMPLOYED/EXT_SOURCE_3            489.5
    AMT_GOODS_PRICE/DAYS_BIRTH            464.5
    DAYS_EMPLOYED/ORGANIZATION_TYPE       460.0
    LIVINGAREA_MEDI                       454.0
    EXT_SOURCE_1/EXT_SOURCE_2             410.0
    EXT_SOURCE_1/OWN_CAR_AGE              405.0
    EXT_SOURCE_1/ORGANIZATION_TYPE        392.0
    DAYS_EMPLOYED/OWN_CAR_AGE             381.5
    sqrt(AMT_GOODS_PRICE)                 380.5
    OWN_CAR_AGE/EXT_SOURCE_2              378.0
    NONLIVINGAREA_MODE                    377.0
    Name: fscore, dtype: float64

保存数据集


```python
df_created.write.bucketBy(100, "SK_ID_CURR").mode("overwrite").saveAsTable("home_credit_default_risk.created_data")
```


```python
spark.stop()
```
