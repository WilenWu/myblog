---
title: PySpark 特征工程(I)--数据预处理
tags:
  - Python
  - Spark
categories:
  - Big Data
  - Spark
cover: /img/spark-install.jpg
top_img: /img/apache-spark-top-img.svg
abbrlink: 59da38ae
description: 
date: 2024-05-28 21:16:02
---

有这么一句话在业界广泛流传：数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。由此可见，特征工程在机器学习中占有相当重要的地位。在实际应用当中，可以说特征工程是机器学习成功的关键。

特征工程是数据分析中最耗时间和精力的一部分工作，它不像算法和模型那样是确定的步骤，更多是工程上的经验和权衡。因此没有统一的方法。这里只是对一些常用的方法做一个总结。

特征工程包含了 Data PreProcessing（数据预处理）、Feature Extraction（特征提取）、Feature Selection（特征选择）和 Feature construction（特征构造）等子问题。

Jupyter Notebook 代码连接：[feature_engineering_on_spark_p1_preproccessing](/ipynb/feature_engineering_on_spark_p1_preproccessing)

# 数据预处理

数据预处理是特征工程的最重要的起始步骤，需要把特征预处理成机器学习模型所能接受的形式。


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

    24/06/01 11:20:13 WARN Utils: Your hostname, MacBook-Air resolves to a loopback address: 127.0.0.1; using 192.168.1.5 instead (on interface en0)
    24/06/01 11:20:13 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    24/06/01 11:20:13 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable


## 探索性数据分析

本项目使用 Kaggle 上的 家庭信用违约风险数据集 (Home Credit Default Risk) ，是一个标准的机器学习分类问题。其目标是使用历史贷款的信息，以及客户的社会经济和财务信息，预测客户是否会违约。

本篇主要通过 application 文件，做基本的数据分析和建模，也是本篇的主要内容。


```python
df = spark.sql("select * from home_credit_default_risk.application_train")
```

    Loading class `com.mysql.jdbc.Driver'. This is deprecated. The new driver class is `com.mysql.cj.jdbc.Driver'. The driver is automatically registered via the SPI and manual loading of the driver class is generally unnecessary.

```python
df.limit(5).toPandas()                                                                 
```


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
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>AMT_ANNUITY</th>
      <th>...</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>191480</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>M</td>
      <td>Y</td>
      <td>N</td>
      <td>0</td>
      <td>157500.0</td>
      <td>342000.0</td>
      <td>17590.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>191502</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>108000.0</td>
      <td>324000.0</td>
      <td>20704.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>191673</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>Y</td>
      <td>Y</td>
      <td>0</td>
      <td>135000.0</td>
      <td>1323000.0</td>
      <td>36513.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>191877</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>2</td>
      <td>45000.0</td>
      <td>47970.0</td>
      <td>5296.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>192108</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>Y</td>
      <td>0</td>
      <td>315000.0</td>
      <td>263686.5</td>
      <td>13522.5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 122 columns</p>
</div>


```python
print(f"dataset shape: ({df.count()}, {len(df.columns)})")
```

    dataset shape: (307511, 122)


```python
# df.printSchema()
```

在遇到非常多的数据的时候，我们一般先会按照数据的类型分布下手，看看不同的数据类型各有多少


```python
# Number of each type of column
dtypes = dict(df.dtypes)
pd.Series(dtypes).value_counts()
```


    double    65
    int       41
    string    16
    Name: count, dtype: int64

接下来看下数据集的统计信息


```python
df.summary().toPandas()
```


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
      <th>summary</th>
      <th>SK_ID_CURR</th>
      <th>TARGET</th>
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>FLAG_OWN_CAR</th>
      <th>FLAG_OWN_REALTY</th>
      <th>CNT_CHILDREN</th>
      <th>AMT_INCOME_TOTAL</th>
      <th>AMT_CREDIT</th>
      <th>...</th>
      <th>FLAG_DOCUMENT_18</th>
      <th>FLAG_DOCUMENT_19</th>
      <th>FLAG_DOCUMENT_20</th>
      <th>FLAG_DOCUMENT_21</th>
      <th>AMT_REQ_CREDIT_BUREAU_HOUR</th>
      <th>AMT_REQ_CREDIT_BUREAU_DAY</th>
      <th>AMT_REQ_CREDIT_BUREAU_WEEK</th>
      <th>AMT_REQ_CREDIT_BUREAU_MON</th>
      <th>AMT_REQ_CREDIT_BUREAU_QRT</th>
      <th>AMT_REQ_CREDIT_BUREAU_YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>count</td>
      <td>307511</td>
      <td>307511</td>
      <td>307511</td>
      <td>307511</td>
      <td>307511</td>
      <td>307511</td>
      <td>307511</td>
      <td>307511</td>
      <td>307511</td>
      <td>...</td>
      <td>307511</td>
      <td>307511</td>
      <td>307511</td>
      <td>307511</td>
      <td>265992</td>
      <td>265992</td>
      <td>265992</td>
      <td>265992</td>
      <td>265992</td>
      <td>265992</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mean</td>
      <td>278180.51857657125</td>
      <td>0.08072881945686496</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.4170517477423572</td>
      <td>168797.91929698447</td>
      <td>599025.9997057016</td>
      <td>...</td>
      <td>0.008129790479039774</td>
      <td>5.951006630657115E-4</td>
      <td>5.072989258920819E-4</td>
      <td>3.349473677364387E-4</td>
      <td>0.006402448193930645</td>
      <td>0.0070002105326475985</td>
      <td>0.0343619356973142</td>
      <td>0.26739526000781977</td>
      <td>0.26547414959848414</td>
      <td>1.899974435321363</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stddev</td>
      <td>102790.17534842461</td>
      <td>0.2724186456483938</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0.722121384437625</td>
      <td>237123.14627885612</td>
      <td>402490.776995855</td>
      <td>...</td>
      <td>0.0897982361093956</td>
      <td>0.024387465065862264</td>
      <td>0.022517620268446132</td>
      <td>0.01829853182243764</td>
      <td>0.08384912844747726</td>
      <td>0.11075740632435459</td>
      <td>0.20468487581282443</td>
      <td>0.9160023961526171</td>
      <td>0.7940556483207575</td>
      <td>1.8692949981815559</td>
    </tr>
    <tr>
      <th>3</th>
      <td>min</td>
      <td>100002</td>
      <td>0</td>
      <td>Cash loans</td>
      <td>F</td>
      <td>N</td>
      <td>N</td>
      <td>0</td>
      <td>25650.0</td>
      <td>45000.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>25%</td>
      <td>189124</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>112500.0</td>
      <td>270000.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>50%</td>
      <td>278173</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>0</td>
      <td>146250.0</td>
      <td>513531.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>75%</td>
      <td>367118</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>1</td>
      <td>202500.0</td>
      <td>808650.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>max</td>
      <td>456255</td>
      <td>1</td>
      <td>Revolving loans</td>
      <td>XNA</td>
      <td>Y</td>
      <td>Y</td>
      <td>19</td>
      <td>1.17E8</td>
      <td>4050000.0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>27.0</td>
      <td>261.0</td>
      <td>25.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 123 columns</p>
</div>

查看目标变量分布


```python
# `TARGET` is the target variable we are trying to predict (0 or 1):
# 1 = Not Repaid 
# 0 = Repaid

# Check if the data is unbalanced
row = df.select(fn.mean('TARGET').alias('rate')).first()
print(f"percentage of default : {row['rate']:.2%}")
df.groupBy("TARGET").count().show() 
```

    percentage of default : 8.07%                                
    
    +------+------+
    |TARGET| count|
    +------+------+
    |     1| 24825|
    |     0|282686|
    +------+------+

## 数据清洗

数据清洗 (Data cleaning) 是对数据进行重新审查和校验的过程，目的在于删除重复信息、纠正存在的错误，并提供数据一致性。

### 数据去重

首先，根据某个 / 多个特征值构成的样本 ID 去重


```python
# `SK_ID_CURR` is the unique id of the row.
df.dropDuplicates(subset=["SK_ID_CURR"]).count() == df.count()
```


    True

### 数据类型转换


```python
dtypes = df.drop("SK_ID_CURR", "TARGET").dtypes

categorical_cols = [k for k, v in dtypes if v == 'string']
numerical_cols = [k for k, v in dtypes if v != 'string']
```

有时，有些数值型特征标识的只是不同类别，其数值的大小并没有实际意义，因此我们将其转化为类别特征。 
本项目并无此类特征，以 hours_appr_process_start 为示例：


```python
# df = df.withColumn('HOUR_APPR_PROCESS_START', df['HOUR_APPR_PROCESS_START'].astype(str))
```

### 错误数据清洗

接下来，我们根据业务常识，或者使用但不限于箱型图（Box-plot）发现数据中不合理的特征值进行清洗。
数据探索时，我们注意到，DAYS_BIRTH列（年龄）中的数字是负数，由于它们是相对于当前贷款申请计算的，所以我们将其转化成正数后查看分布


```python
df.select(df['DAYS_BIRTH'] / -365).summary().show()
```

    +-------+-------------------+
    |summary|(DAYS_BIRTH / -365)|
    +-------+-------------------+
    |  count|             307511|
    |   mean|  43.93697278587162|
    | stddev| 11.956133237768654|
    |    min| 20.517808219178082|
    |    25%|  34.00547945205479|
    |    50%|  43.14794520547945|
    |    75%| 53.917808219178085|
    |    max|  69.12054794520547|
    +-------+-------------------+

那些年龄看起来合理，没有异常值。
接下来，我们对其他的 DAYS 特征作同样的分析


```python
for feature in ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']:
        print(f'{feature} info: ')
        df.select(df[feature] / -365).summary().show()                    
```

    DAYS_BIRTH info:   
    +-------+-------------------+
    |summary|(DAYS_BIRTH / -365)|
    +-------+-------------------+
    |  count|             307511|
    |   mean|  43.93697278587162|
    | stddev| 11.956133237768654|
    |    min| 20.517808219178082|
    |    25%|  34.00547945205479|
    |    50%|  43.14794520547945|
    |    75%| 53.917808219178085|
    |    max|  69.12054794520547|
    +-------+-------------------+
    
    DAYS_EMPLOYED info: 
    +-------+----------------------+
    |summary|(DAYS_EMPLOYED / -365)|
    +-------+----------------------+
    |  count|                307511|
    |   mean|   -174.83574220287002|
    | stddev|    387.05689457185537|
    |    min|   -1000.6657534246575|
    |    25%|    0.7917808219178082|
    |    50%|    3.3232876712328765|
    |    75%|     7.558904109589041|
    |    max|     49.07397260273972|
    +-------+----------------------+
    
    DAYS_REGISTRATION info: 
    +-------+--------------------------+
    |summary|(DAYS_REGISTRATION / -365)|
    +-------+--------------------------+
    |  count|                    307511|
    |   mean|        13.660603637091562|
    | stddev|         9.651743345104306|
    |    min|                      -0.0|
    |    25%|         5.504109589041096|
    |    50%|        12.336986301369864|
    |    75%|        20.487671232876714|
    |    max|         67.59452054794521|
    +-------+--------------------------+
    
    DAYS_ID_PUBLISH info: 
    +-------+------------------------+
    |summary|(DAYS_ID_PUBLISH / -365)|
    +-------+------------------------+
    |  count|                  307511|
    |   mean|        8.20329417328335|
    | stddev|       4.135480600008283|
    |    min|                    -0.0|
    |    25%|      4.7095890410958905|
    |    50%|       8.915068493150685|
    |    75%|      11.775342465753425|
    |    max|       19.71780821917808|
    +-------+------------------------+                                                    


```python
buckets = df.select((df['DAYS_EMPLOYED'] / -365).alias('DAYS_EMPLOYED'))

bucketizer = ft.QuantileDiscretizer(numBuckets=10, inputCol='DAYS_EMPLOYED', outputCol='buckets').fit(buckets)
buckets = bucketizer.transform(buckets)

buckets.groupBy('buckets').count().sort('buckets').show()
bucketizer.getSplits()
```

    +-------+-----+
    |buckets|count|
    +-------+-----+
    |    1.0|61425|
    |    2.0|30699|
    |    3.0|30733|
    |    4.0|30685|
    |    5.0|30741|
    |    6.0|30716|
    |    7.0|30750|
    |    8.0|30726|
    |    9.0|31036|
    +-------+-----+


    [-inf,
     -1000.6657534246575,
     0.39452054794520547,
     1.252054794520548,
     2.2465753424657535,
     3.317808219178082,
     4.635616438356164,
     6.457534246575342,
     8.827397260273973,
     13.2986301369863,
     inf]

有超过60000个用户的DAYS_EMPLOYED在1000年上，可以猜测这只是缺失值标记。


```python
# Replace the anomalous values with nan
df_emp = df.select(fn.when(df['DAYS_EMPLOYED']>=365243, None).otherwise(df['DAYS_EMPLOYED']).alias('DAYS_EMPLOYED'))

df_emp.sample(0.1).toPandas().plot.hist(title = 'Days Employment Histogram')
plt.xlabel('Days Employment')
```


![](/img/feature_engineering_on_spark/preproccessing_output_25_2.png)


可以看到，数据分布基本正常了。   

### 布尔特征清洗


```python
for col in categorical_cols:
    unique_count = df.select(col).dropna().distinct().count()
    if unique_count == 2:
        df.groupBy(col).count().show()                          
```

    +------------------+------+
    |NAME_CONTRACT_TYPE| count|
    +------------------+------+
    |   Revolving loans| 29279|
    |        Cash loans|278232|
    +------------------+------+
    +------------+------+
    |FLAG_OWN_CAR| count|
    +------------+------+
    |           Y|104587|
    |           N|202924|
    +------------+------+
    +---------------+------+
    |FLAG_OWN_REALTY| count|
    +---------------+------+
    |              Y|213312|
    |              N| 94199|
    +---------------+------+
    +-------------------+------+
    |EMERGENCYSTATE_MODE| count|
    +-------------------+------+
    |               NULL|145755|
    |                 No|159428|
    |                Yes|  2328|
    +-------------------+------+                                                                              


```python
cols_to_transform = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'EMERGENCYSTATE_MODE']
df.replace(
    ['Y', 'N', 'Yes', 'No'], ['1', '0', '1', '0'], 
    subset=cols_to_transform
).select(cols_to_transform).show(5)
```

    +------------+---------------+-------------------+
    |FLAG_OWN_CAR|FLAG_OWN_REALTY|EMERGENCYSTATE_MODE|
    +------------+---------------+-------------------+
    |           1|              0|                  0|
    |           0|              1|                  0|
    |           1|              1|               NULL|
    |           0|              1|               NULL|
    |           0|              1|                  0|
    +------------+---------------+-------------------+
    only showing top 5 rows

### 函数封装

最后，使用函数封装以上步骤：


```python
dtypes = df.drop("SK_ID_CURR", "TARGET").dtypes
categorical_cols = [k for k, v in dtypes if v == 'string']
numerical_cols = [k for k, v in dtypes if v != 'string']

# Data cleaning
def clean(df):
    # remove duplicates.
    df = df.dropDuplicates(subset=["SK_ID_CURR"])
    
    # transform
    cols_to_transform = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'EMERGENCYSTATE_MODE']
    df = df.replace(
        ['Y', 'N', 'Yes', 'No'], ['1', '0', '1', '0'], 
        subset=cols_to_transform
    )
    df = df.withColumns({c: df[c].cast('int') for c in cols_to_transform})
    
    # Replace the anomalous values with nan
    df = df.withColumn('DAYS_EMPLOYED', 
        fn.when(df['DAYS_EMPLOYED']>=365243, None).otherwise(df['DAYS_EMPLOYED'])
    )
    
    df = df.replace('XNA', None)
    df = df.withColumnRenamed("TARGET", "label")
    return df

df = clean(df)
```

## 特征重编码

有很多机器学习算法只能接受数值型特征的输入，不能处理离散值特征，比如线性回归，逻辑回归等线性模型，那么我们需要将离散特征重编码成数值变量。

现在我们来看看每个分类特征的类别数：


```python
df.select([fn.countDistinct(col).alias(col) for col in categorical_cols]).show(1, vertical=True)
```

    -RECORD 0-------------------------
     NAME_CONTRACT_TYPE         | 2   
     CODE_GENDER                | 2   
     FLAG_OWN_CAR               | 2   
     FLAG_OWN_REALTY            | 2   
     NAME_TYPE_SUITE            | 7   
     NAME_INCOME_TYPE           | 8   
     NAME_EDUCATION_TYPE        | 5   
     NAME_FAMILY_STATUS         | 6   
     NAME_HOUSING_TYPE          | 6   
     OCCUPATION_TYPE            | 18  
     WEEKDAY_APPR_PROCESS_START | 7   
     ORGANIZATION_TYPE          | 57  
     FONDKAPREMONT_MODE         | 4   
     HOUSETYPE_MODE             | 3   
     WALLSMATERIAL_MODE         | 7   
     EMERGENCYSTATE_MODE        | 2                         

1. 变量 NAME_EDUCATION_TYPE 表征着潜在的排序关系，可以使用顺序编码。
2. 变量 OCCUPATION_TYPE （职业类型）和 ORGANIZATION_TYPE 类别数较多，准备使用平均数编码。
3. 剩余的无序分类特征使用one-hot编码。

### 顺序编码

**有序分类特征**实际上表征着潜在的排序关系，我们将这些特征的类别映射成有大小的数字，因此可以用顺序编码。   

让我们从分类特征中手动提取有序级别：


```python
# The ordinal (ordered) categorical features
# Pandas calls the categories "levels"

ordered_levels = {
    "NAME_EDUCATION_TYPE": ["Lower secondary", 
                            "Secondary / secondary special", 
                            "Incomplete higher", 
                            "Higher education"]
}
```

spark中的StringIndexer是按特征值出现的频率编码，我们需要自定义一个编码函数。


```python
def ordinal_encode(df, levels):
    for var, to_replace in levels.items():
        mapping = {v: str(i) for i,v in enumerate(to_replace)}
        df = df.replace(mapping, subset=[var])
        df = df.withColumn(var, df[var].cast('int'))
    print(f'{len(levels):d} columns were ordinal encoded')
    return df
```


```python
ordinal_encode(df, ordered_levels).groupBy(*ordered_levels.keys()).count().show()
```

    1 columns were ordinal encoded
    +-------------------+------+
    |NAME_EDUCATION_TYPE| count|
    +-------------------+------+
    |               NULL|   164|
    |                  1|218391|
    |                  3| 74863|
    |                  2| 10277|
    |                  0|  3816|
    +-------------------+------+                                                        

### 平均数编码

一般情况下，针对分类特征，我们只需要OneHotEncoder或OrdinalEncoder进行编码，这类简单的预处理能够满足大多数数据挖掘算法的需求。如果某一个分类特征的可能值非常多（高基数 high cardinality），那么再使用one-hot编码往往会出现维度爆炸。平均数编码（mean encoding）是一种高效的编码方式，在实际应用中，能极大提升模型的性能。

变量 OCCUPATION_TYPE （职业类型）和 ORGANIZATION_TYPE类别数较多，准备使用平均数编码。


```python
class MeanEncoder(Estimator, Transformer):
    def __init__(self, smoothing=0.0, inputCols=None, labelCol="label"):
        """
        The MeanEncoder() replaces categories by the mean value of the target for each
        category.
        
        math:
            mapping = (w_i) posterior + (1-w_i) prior
        where
            w_i = n_i t / (s + n_i t)
        
        In the previous equation, t is the target variance in the entire dataset, s is the
        target variance within the category and n is the number of observations for the
        category.
        
        Parameters
        ----------
        smoothing: int, float, 'auto', default=0.0
        """
        super().__init__()
        self.smoothing = smoothing
        self.inputCols = inputCols
        self.labelCol = labelCol
    
    def _fit(self, df):
        """
        Learn the mean value of the target for each category of the variable.
        """

        self.encoder_dict = {}
        inputCols = self.inputCols
        labelCol = self.labelCol
        y_prior = df.select(fn.mean(labelCol).alias("mean")).first()["mean"]
        
        for var in inputCols:
            if self.smoothing == "auto":
                y_var = df.cov(labelCol, labelCol)
                damping = fn.variance(labelCol) / y_var
            else:
                damping = fn.lit(self.smoothing)
            
            groups = df.groupBy(var).agg(
                fn.mean(labelCol).alias("posterior"),
                fn.count("*").alias("counts"),
                damping.alias("damping") 
            ).toPandas().dropna()
            
            groups["lambda"] = groups["counts"] / (groups["counts"] + groups["damping"])
            groups["code"] = (
                groups["lambda"] * groups["posterior"] + 
                    (1.0 - groups["lambda"]) * y_prior
            )
            
            self.encoder_dict[var] = dict(zip(groups[var], groups["code"]))
        return self
    
    def _transform(self, df):
        for var in self.encoder_dict:
            mapping = {k: str(v) for k,v in self.encoder_dict[var].items()}
            df = df.replace(mapping, subset=[var])
            df = df.withColumn(var, df[var].cast('float'))

        print(f'{len(self.encoder_dict):d} columns were mean encoded')
        return df
```


```python
# replace categories by the mean value of the target for each category.
inputCols = ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
mean_encoder = MeanEncoder(
    inputCols=inputCols, 
    labelCol='label',
    smoothing='auto'
)
mean_encoder.fit(df).transform(df).select(inputCols).show(5)  
```

    2 columns were mean encoded
    +---------------+-----------------+
    |OCCUPATION_TYPE|ORGANIZATION_TYPE|
    +---------------+-----------------+
    |    0.062140968|       0.09299603|
    |     0.09631742|       0.09449421|
    |    0.113258936|       0.10173836|
    |           NULL|             NULL|
    |           NULL|             NULL|
    +---------------+-----------------+
    only showing top 5 rows

### 哑变量编码

**无序分类特征**对于树集成模型（tree-ensemble like XGBoost）是可用的，但对于线性模型（like Lasso or Ridge）则必须使用one-hot重编码。接下来我们把上节索引化的无序分类特征进行编码。


```python
# The nominative (unordered) categorical features
encoded_cols = ['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
nominal_categories = [col for col in categorical_cols if col not in encoded_cols]

indexedCols = [f"indexed_{col}" for col in nominal_categories]
vectorCols = [f"encoded_{col}" for col in nominal_categories]

onehot_encoder = Pipeline(stages=[
    StringIndexer(
        inputCols=nominal_categories, 
        outputCols=indexedCols,
        handleInvalid='keep'
    ),
    OneHotEncoder(
        inputCols=indexedCols,
        outputCols=vectorCols
    )
])
onehot_encoder.fit(df).transform(df).select(vectorCols).limit(5).toPandas()
```


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
      <th>encoded_NAME_CONTRACT_TYPE</th>
      <th>encoded_CODE_GENDER</th>
      <th>encoded_FLAG_OWN_CAR</th>
      <th>encoded_FLAG_OWN_REALTY</th>
      <th>encoded_NAME_TYPE_SUITE</th>
      <th>encoded_NAME_INCOME_TYPE</th>
      <th>encoded_NAME_FAMILY_STATUS</th>
      <th>encoded_NAME_HOUSING_TYPE</th>
      <th>encoded_WEEKDAY_APPR_PROCESS_START</th>
      <th>encoded_FONDKAPREMONT_MODE</th>
      <th>encoded_HOUSETYPE_MODE</th>
      <th>encoded_WALLSMATERIAL_MODE</th>
      <th>encoded_EMERGENCYSTATE_MODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(0.0, 1.0)</td>
      <td>(0.0, 1.0)</td>
      <td>(0.0, 1.0)</td>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0, 1.0)</td>
      <td>(1.0, 0.0, 0.0)</td>
      <td>(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(1.0, 0.0)</td>
      <td>(0.0, 1.0)</td>
      <td>(0.0, 1.0)</td>
      <td>(1.0, 0.0)</td>
      <td>(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0)</td>
      <td>(0.0, 1.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0)</td>
      <td>(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0)</td>
    </tr>
  </tbody>
</table>
</div>

### 连续特征分箱

Binning Continuous Features

在实际的模型训练过程中，我们也经常对连续特征进行离散化处理，这样能消除特征量纲的影响，同时还能极大减少异常值的影响，增加特征的稳定性。

分箱主要分为等频分箱、等宽分箱和聚类分箱三种。等频分箱会一定程度受到异常值的影响，而等宽分箱又容易完全忽略异常值信息，从而一定程度上导致信息损失，若要更好的兼顾变量的原始分布，则可以考虑聚类分箱。所谓聚类分箱，指的是先对某连续变量进行聚类（往往是 k-Means 聚类），然后使用样本所属类别。

以年龄对还款的影响为例


```python
# Find the correlation of the positive days since birth and target
df.select((df['DAYS_BIRTH'] / -365).alias('age'), 'label').corr('age', "label")
```


    -0.07823930830982699

可见，客户年龄与目标意义呈负相关关系，即随着客户年龄的增长，他们往往会更经常地按时偿还贷款。我们接下来将制作一个核心密度估计图（KDE），直观地观察年龄对目标的影响。 


```python
sample = df.sample(0.1).select((df['DAYS_BIRTH']/fn.lit(-365)).alias("age"), "label").toPandas()

plt.figure(figsize = (5, 3))
sns.kdeplot(data=sample, x="age", hue="label", common_norm=False)
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Distribution of Ages')
```


![](/img/feature_engineering_on_spark/preproccessing_output_48_2.png)


如果我们把年龄分箱：


```python
# Bin the age data
age_binned = pd.cut(sample['age'], bins = np.linspace(20, 70, num = 11))
age_groups  = sample['label'].groupby(age_binned).mean()

plt.figure(figsize = (8, 3))
# Graph the age bins and the average of the target as a bar plot
sns.barplot(x=age_groups.index, y=age_groups*100)
# Plot labeling
plt.xticks(rotation = 30)
plt.xlabel('Age Group (years)')
plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group')
```

![](/img/feature_engineering_on_spark/preproccessing_output_50_1.png)

有一个明显的趋势：年轻的申请人更有可能不偿还贷款！ 年龄最小的三个年龄组的失败率在10％以上，最老的年龄组为5％。   
pyspark.ml.feature 模块中的 Bucketizer 可以实现等宽分箱，QuantileDiscretizer可以实现等频分箱。


```python
bucketizer = ft.QuantileDiscretizer(
    numBuckets=10,
    handleInvalid='keep',
    inputCols=['DAYS_BIRTH', 'DAYS_EMPLOYED'], 
    outputCols=["buckets1", "buckets2"]
).fit(df)

splits = bucketizer.getSplitsArray() # bin_edges
for c, s in zip(['DAYS_BIRTH', 'DAYS_EMPLOYED'], splits):
    print(f"{c}'s bin_edges:")
    print(s)
```

    DAYS_BIRTH's bin_edges:
    [-inf, -22185.0, -20480.0, -18892.0, -17228.0, -15759.0, -14425.0, -13153.0, -11706.0, -10296.0, inf]
    DAYS_EMPLOYED's bin_edges:
    [-inf, -5338.0, -3679.0, -2795.0, -2164.0, -1650.0, -1253.0, -922.0, -619.0, -336.0, inf] 

### 函数封装


```python
dtypes = df.drop("SK_ID_CURR", "TARGET").dtypes
categorical_cols = [k for k, v in dtypes if v == 'string']
numerical_cols = [k for k, v in dtypes if v != 'string']   

def encode(df):
    # The ordinal (ordered) categorical features
    # Pandas calls the categories "levels"
    ordered_levels = {
        "NAME_EDUCATION_TYPE": ["Lower secondary", 
                                "Secondary / secondary special", 
                                "Incomplete higher", 
                                "Higher education"]
    }
    df = ordinal_encode(df, ordered_levels)
    
    # replace categories by the mean value of the target for each category.
    mean_encoder = MeanEncoder(
        inputCols=['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'], 
        labelCol='label',
        smoothing='auto'
    )
    df = mean_encoder.fit(df).transform(df)
    
    # The nominative (unordered) categorical features
    nominal_categories = [col for col in categorical_cols if col not in ordered_levels]
    features_onehot = [col for col in nominal_categories if col not in ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']]

    indexedCols = [f"indexed_{col}" for col in features_onehot]
    encodedCols = [f"encoded_{col}" for col in features_onehot]

    onehot_encoder = Pipeline(stages=[
        StringIndexer(
            inputCols=features_onehot, 
            outputCols=indexedCols,
            handleInvalid='keep'
        ),
        OneHotEncoder(
            inputCols=indexedCols,
            outputCols=encodedCols
        )
    ])
    
    df = onehot_encoder.fit(df).transform(df).drop(*features_onehot + indexedCols)
    print(f'{len(features_onehot):d} columns were one-hot encoded')
    
    colsMap = dict(zip(encodedCols, features_onehot))
    df = df.withColumnsRenamed(colsMap)
    return df
```


```python
# Encode categorical features
df_encoded = encode(df)
df_encoded.select(categorical_cols).limit(5).toPandas()
```

    1 columns were ordinal encoded
    2 columns were mean encoded
    10 columns were one-hot encoded

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
      <th>NAME_CONTRACT_TYPE</th>
      <th>CODE_GENDER</th>
      <th>NAME_TYPE_SUITE</th>
      <th>NAME_INCOME_TYPE</th>
      <th>NAME_EDUCATION_TYPE</th>
      <th>NAME_FAMILY_STATUS</th>
      <th>NAME_HOUSING_TYPE</th>
      <th>OCCUPATION_TYPE</th>
      <th>WEEKDAY_APPR_PROCESS_START</th>
      <th>ORGANIZATION_TYPE</th>
      <th>FONDKAPREMONT_MODE</th>
      <th>HOUSETYPE_MODE</th>
      <th>WALLSMATERIAL_MODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(0.0, 1.0)</td>
      <td>(0.0, 1.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>3</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>0.062141</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>0.092996</td>
      <td>(0.0, 0.0, 0.0, 1.0)</td>
      <td>(1.0, 0.0, 0.0)</td>
      <td>(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>2</td>
      <td>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>0.096317</td>
      <td>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)</td>
      <td>0.094494</td>
      <td>(1.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(1.0, 0.0)</td>
      <td>(0.0, 1.0)</td>
      <td>(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>1</td>
      <td>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>0.113259</td>
      <td>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>0.101738</td>
      <td>(0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>1</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>NaN</td>
      <td>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)</td>
      <td>NaN</td>
      <td>(0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(1.0, 0.0)</td>
      <td>(1.0, 0.0)</td>
      <td>(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>1</td>
      <td>(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)</td>
      <td>(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>NaN</td>
      <td>(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)</td>
      <td>NaN</td>
      <td>(0.0, 0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0)</td>
      <td>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)</td>
    </tr>
  </tbody>
</table>
</div>


```python
pd.Series(dict(df_encoded.dtypes)).value_counts()
```


    double    65
    int       45
    vector    10
    float      2
    Name: count, dtype: int64

## 缺失值处理

特征有缺失值是非常常见的，大部分机器学习模型在拟合前需要处理缺失值（Handle Missing Values）。

### 缺失值统计


```python
# Function to calculate missing values by column
def display_missing(df, threshold=None, verbose=1):
    n = df.count()
    exprs = [fn.sum(df[col].isNull().cast('int')).alias(col) for col in df.columns]
    missing_number = df.select(*exprs).first().asDict()
    missing_df = pd.DataFrame({
        "missing_number": missing_number.values(),  # Total missing values
        "missing_rate": [value / n for value in missing_number.values()]   # Proportion of missing values
        }, index=missing_number.keys())
    missing_df = missing_df.query("missing_rate>0").sort_values("missing_rate", ascending=False)
    threshold = 0.25 if threshold is None else threshold
    high_missing = missing_df.query(f"missing_rate>{threshold}")
    # Print some summary information
    if verbose:
        print(f"Your selected dataframe has {missing_df.shape[0]} out of {len(df.columns)} columns that have missing values.")
    # Return the dataframe with missing information
    if threshold is None:
        return missing_df
    else:
        if verbose:
            print(f"There are {high_missing.shape[0]} columns with more than {threshold:.1%} missing values.")
        return high_missing
```


```python
# Missing values statistics
print(display_missing(df_encoded).head(10))
```

    Your selected dataframe has 66 out of 122 columns that have missing values.
    There are 47 columns with more than 25.0% missing values.
                              missing_number  missing_rate
    COMMONAREA_MEDI                   214865      0.698723
    COMMONAREA_MODE                   214865      0.698723
    COMMONAREA_AVG                    214865      0.698723
    NONLIVINGAPARTMENTS_MODE          213514      0.694330
    NONLIVINGAPARTMENTS_MEDI          213514      0.694330
    NONLIVINGAPARTMENTS_AVG           213514      0.694330
    LIVINGAPARTMENTS_MODE             210199      0.683550
    LIVINGAPARTMENTS_MEDI             210199      0.683550
    LIVINGAPARTMENTS_AVG              210199      0.683550
    FLOORSMIN_MODE                    208642      0.678486                                       

### 缺失值删除 

如果某个特征的缺失值超过阈值（例如80%），那么该特征对模型的贡献就会降低，通常就可以考虑删除该特征。


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
```


```python
_ = drop_missing_data(df_encoded, threshold=0.2)
```

    Removed 0 variables with missing more than 80.0%

### 缺失值标记

有时，对于每个含有缺失值的列，我们额外添加一列来表示该列中缺失值的位置，在某些应用中，能取得不错的效果。
继续分析之前清洗过的 DAYS_EMPLOYED 异常，我们对缺失数据进行标记，看看他们是否影响客户违约。


```python
df_encoded.groupBy(df_encoded['DAYS_EMPLOYED'].isNull()).mean('label').show()
```

    +-----------------------+-------------------+
    |(DAYS_EMPLOYED IS NULL)|         avg(label)|
    +-----------------------+-------------------+
    |                   true|0.05399646043269404|
    |                  false| 0.0865997453765215|
    +-----------------------+-------------------+

发现缺失值的逾期率 5.4% 低于正常值的逾期率 8.66%，与Target的相关性很强，因此新增一列DAYS_EMPLOYED_MISSING 标记。这种处理对线性方法比较有效，而基于树的方法可以自动识别。


```python
# Adds a binary variable to flag missing observations.
from pyspark.ml.stat import Correlation, ChiSquareTest

def flag_missing(df, inputCols=None, labelCol='label', alpha=0.05):
    """
    Adds a binary variable to flag missing observations(one indicator per variable). 
    The added variables (missing indicators) are named with the original variable name plus '_missing'.
    
    Parameters:
    ----------
    alpha: float, default=0.05
        Features with correlation more than alpha are selected.
    """
    if inputCols is None:
        inputCols = df.drop(labelCol).columns
    
    for var in inputCols:
        df = df.withColumn(var + "_missing", df[var].isNull().cast('int'))
    
    indicators = [var + "_missing" for var in inputCols]
    # The correlations
    corr = df.select([fn.corr(labelCol, c2).alias(c2) for c2 in indicators])
    corr = corr.fillna(0).first().asDict()
    # find variables for which indicator should be added.
    selected_cols = [var for var, r in corr.items() if abs(r) > alpha]
    drop_cols = [var for var in indicators if var not in selected_cols]
    df = df.drop(*drop_cols)
    print(f"Added {len(selected_cols)} missing indicators")
    return df
```


```python
print('The number of features:', len(flag_missing(df_encoded).columns))
```

    Added 0 missing indicators
    The number of features: 122

### 人工插补

根据业务知识来进行人工填充。 

若变量是类别型，且不同值较少，可在编码时转换成哑变量。例如，编码前的性别变量 code_gender


```python
pipeline = Pipeline(stages=[
    StringIndexer(
        inputCol="CODE_GENDER", 
        outputCol="indexedCol",
        handleInvalid="keep"
    ),
    OneHotEncoder(
        inputCol="indexedCol", 
        outputCol="encodedCol", 
        handleInvalid="keep",
        dropLast=False
    )
])

pipeline.fit(df).transform(df).select("CODE_GENDER", "encodedCol").show(5)                            
```

    +-----------+-------------+
    |CODE_GENDER|   encodedCol|
    +-----------+-------------+
    |          M|(4,[1],[1.0])|
    |          F|(4,[0],[1.0])|
    |          M|(4,[1],[1.0])|
    |          F|(4,[0],[1.0])|
    |          F|(4,[0],[1.0])|
    +-----------+-------------+
    only showing top 5 rows

分类特征在索引化时已经处理了缺失值，因此不需要再特殊处理。   
若变量是布尔型，视情况可统一填充为零


```python
nunique = df_encoded.select([fn.countDistinct(var).alias(var) for var in df_encoded.columns]).first().asDict() 
binary = df_encoded.select([fn.collect_set(var).alias(var) for var,n in nunique.items() if n == 2])
print([k for k, v in binary.first().asDict().items() if set(v) == {0, 1}])
```

    ['label', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'EMERGENCYSTATE_MODE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']


 如果我们仔细观察一下字段描述，会发现很多缺失值都有迹可循，比如客户的社会关系中有30天/60天逾期及申请贷款前1小时/天/周/月/季度/年查询了多少次征信的都可填充为数字0。


```python
def impute_manually(df):
    """
    Replaces missing values by an arbitrary value
    """
    # boolean
    boolean_features = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 
                        'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 
                        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
                        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 
                        'EMERGENCYSTATE_MODE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 
                        'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 
                        'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 
                        'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 
                        'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
    df = df.na.fill(0, subset=boolean_features)
    # fill 0
    features_fill_zero = [
        "OBS_30_CNT_SOCIAL_CIRCLE",  
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR"
    ]
    df = df.na.fill(0, subset=features_fill_zero)
    
    return df
```


```python
_ = display_missing(impute_manually(df_encoded))
```

    Your selected dataframe has 55 out of 122 columns that have missing values.
    There are 46 columns with more than 25.0% missing values.

### 条件平均值填充法

通过之前的相关分析，我们知道AMT_ANNUITY这个特征与AMT_CREDIT和AMT_INCOME_TOTAL有比较大的关系，所以这里用这两个特征分组后的中位数进行插补，称为条件平均值填充法（Conditional Mean Completer）。


```python
print('AMT_CREDIT :', df.corr('AMT_CREDIT', 'AMT_ANNUITY'))
print('AMT_INCOME_TOTAL :', df.corr('AMT_CREDIT', 'AMT_ANNUITY'))  
```

    AMT_CREDIT : 0.7700800319525184
    AMT_INCOME_TOTAL : 0.7700800319525184


```python
# conditional statistic completer
class ConditionalMeanCompleter:
    pass
```

### 简单插补

 `Imputer` 支持平均值、中位数或众数插补缺失值，目前不支持分类特征。


```python
# Univariate imputer for completing missing values with simple strategies.

dtypes = df_encoded.drop("SK_ID_CURR", "TARGET").dtypes
numerical_cols = [k for k, v in dtypes if v not in ('string', 'vector')]
imputed_cols = [f"imputed_{col}" for col in numerical_cols]
imputer = ft.Imputer(
    inputCols=numerical_cols,
    outputCols=imputed_cols,
    strategy="median"
)
```


```python
_ = display_missing(imputer.fit(df_encoded).transform(df_encoded).select(imputed_cols))
```

    Your selected dataframe has 0 out of 111 columns that have missing values.
    There are 0 columns with more than 25.0% missing values.

### 函数封装

最后，总结下我们的缺失处理策略：

- 删除缺失率高于80%特征
- 添加缺失标记
- 有业务含义的进行人工插补
- 最后简单统计插补


```python
# Function for missing value imputation

def handle_missing(df):
    # Remove variables with high missing rate
    df = drop_missing_data(df, threshold=0.2)
    # find variables for which indicator should be added.
    df = flag_missing(df)

    # Replaces missing values by an arbitrary value
    df = impute_manually(df)

    # Univariate imputer for completing missing values with simple strategies.
    dtypes = df.drop("SK_ID_CURR", "TARGET").dtypes
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
df_imputed = handle_missing(df_encoded)
```

    Removed 0 variables with missing more than 80.0%
    Added 0 missing indicators                  

确认缺失值是否已全部处理完毕：


```python
_ = display_missing(df_imputed)
```

    Your selected dataframe has 0 out of 122 columns that have missing values.
    There are 0 columns with more than 25.0% missing values.

## 异常值检测

我们在实际项目中拿到的数据往往有不少异常数据，这些异常数据很可能让我们模型有很大的偏差。异常检测的方法有很多，例如3倍标准差、箱线法的单变量标记，或者聚类、iForest和LocalOutlierFactor等无监督学习方法。

- **箱线图检测**根据四分位点判断是否异常。四分位数具有鲁棒性，不受异常值的干扰。通常认为小于 $Q_1-1.5*IQR$ 或大于 $Q_3+1.5*IQR$ 的点为离群点。 
- **3倍标准差原则**：假设数据满足正态分布，通常定义偏离均值的 $3\sigma$ 之外内的点为离群点，$\mathbb P(|X-\mu|<3\sigma)=99.73\%$​。如果数据不服从正态分布，也可以用远离平均值的多少倍标准差来描述。

筛选出来的异常样本需要根据实际含义处理：

- 根据异常点的数量和影响，考虑是否将该条记录删除。
- 对数据做 log-scale 变换后消除异常值。
- 通过数据分箱来平滑异常值。
- 使用均值/中位数/众数来修正替代异常点，简单高效。
- 标记异常值或新增异常值得分列。
- 树模型对离群点的鲁棒性较高，可以选择忽略异常值。


```python
class OutlierCapper(Estimator, Transformer):
    """
    Caps maximum and/or minimum values of a variable at automatically
    determined values.
    Works only with numerical variables. A list of variables can be indicated. 
    
    Parameters
    ----------
    method: str, 'gaussian' or 'iqr', default='iqr'
        If method='gaussian': 
            - upper limit: mean + 3 * std
            - lower limit: mean - 3 * std
        If method='iqr': 
            - upper limit: 75th quantile + 3 * IQR
            - lower limit: 25th quantile - 3 * IQR
            where IQR is the inter-quartile range: 75th quantile - 25th quantile.
    fold: int, default=3   
        You can select how far out to cap the maximum or minimum values.
    """

    def __init__(self, inputCols, method='iqr', fold=3):
        super().__init__()
        self.method = method
        self.fold = fold
        self.inputCols = inputCols

    def _fit(self, df):
        """
        Learn the values that should be used to replace outliers.
        """

        if self.method == "gaussian":
            mean = df.select([fn.mean(var).alias(var) for var in self.inputCols])
            mean = pd.Series(mean.first().asDict())
            bias= [mean, mean]
            scale = df.select([fn.std(var).alias(var) for var in self.inputCols])
            scale = pd.Series(scale.first().asDict())
        elif self.method == "iqr":
            Q1 = df.select([fn.percentile(var, 0.25).alias(var) for var in self.inputCols])
            Q1 = pd.Series(Q1.first().asDict())
            Q3 = df.select([fn.percentile(var, 0.75).alias(var) for var in self.inputCols])
            Q3 = pd.Series(Q3.first().asDict())
            bias = [Q1, Q3]
            scale = Q3 - Q1         
        
        # estimate the end values
        if (scale == 0).any():
            raise ValueError(
                f"Input columns {scale[scale == 0].index.tolist()!r}"
                f" have low variation for method {self.method!r}."
                f" Try other capping methods or drop these columns."
            )
        else:
            self.upper_limit = bias[1] + self.fold * scale
            self.lower_limit = bias[0] - self.fold * scale  

        return self 

    def _transform(self, df):
        """
        Cap the variable values.
        """
        maximum = df.select([fn.max(var).alias(var) for var in self.inputCols])
        maximum = pd.Series(maximum.first().asDict())
        minimum = df.select([fn.min(var).alias(var) for var in self.inputCols])
        minimum = pd.Series(minimum.first().asDict())
        outiers = (maximum.gt(self.upper_limit) | 
                   minimum.lt(self.lower_limit))
        n = outiers.sum()
        print(f"Your selected dataframe has {n} out of {len(self.inputCols)} columns that have outliers.")
        
        # replace outliers
        for var in self.inputCols:
            upper_limit = self.upper_limit[var]
            lower_limit = self.lower_limit[var]
            df = df.withColumn(var, 
                fn.when(df[var] > upper_limit, upper_limit)
                  .when(df[var] < lower_limit, lower_limit)
                  .otherwise(df[var])
            )
        return df
```


```python
outlier_capper = OutlierCapper(method="gaussian", inputCols=numerical_cols).fit(df_imputed)
df_capped = outlier_capper.transform(df_imputed)
```

    Your selected dataframe has 96 out of 111 columns that have outliers.


## 标准化/归一化

数据标准化和归一化可以提高一些算法的准确度，也能加速梯度下降收敛速度。也有不少模型不需要做标准化和归一化，主要是基于概率分布的模型，比如决策树大家族的CART，随机森林等。

- **z-score标准化**是最常见的特征预处理方式，基本所有的线性模型在拟合的时候都会做标准化。前提是假设特征服从正态分布，标准化后，其转换成均值为0标准差为1的标准正态分布。
- **max-min标准化**也称为离差标准化，预处理后使特征值映射到[0,1]之间。这种方法的问题就是如果测试集或者预测数据里的特征有小于min，或者大于max的数据，会导致max和min发生变化，需要重新计算。所以实际算法中， 除非你对特征的取值区间有需求，否则max-min标准化没有 z-score标准化好用。
- **L1/L2范数标准化**：如果我们只是为了统一量纲，那么通过L2范数整体标准化。

| pyspark.ml.feature                          |                                        标准化 |
| ------------------------------------------- | ----------------------------------------------- |
| StandardScaler(withMean, withStd, …)        | 是一个`Estimator`。z-scoe标准化                              |
| Normalizer(p, inputCol, outputCol)          | 是一个`Transformer`。该方法使用p范数将数据缩放为单位范数（默认为L2） |
| MaxAbsScaler(inputCol, outputCol)           | 是一个`Estimator`。将数据标准化到`[-1, 1]`范围内             |
| MinMaxScaler(min, max, inputCol, outputCol) | 是一个`Estimator`。将数据标准化到`[0, 1]`范围内              |
| RobustScaler(lower, upper, …)               | 是一个`Estimator`。根据分位数缩放数据                        |

由于数据集中依然存在一定的离群点，我们可以用RobustScaler对数据进行标准化处理。


```python
from pyspark.ml.feature import RobustScaler

scaler = RobustScaler(inputCol="features", outputCol="scaled")
assembler = VectorAssembler(
    inputCols=['DAYS_EMPLOYED', 'AMT_CREDIT'],
    outputCol="features"
)
pipelineModel = Pipeline(stages=[assembler, scaler]).fit(df_imputed)
pipelineModel.transform(df_imputed).select('scaled').show(5)
```

    +--------------------+
    |              scaled|
    +--------------------+
    |[-0.9644030668127...|
    |[-0.5991237677984...|
    |[-0.6056955093099...|
    |[-0.9036144578313...|
    |[-0.9036144578313...|
    +--------------------+
    only showing top 5 rows

## 正态变换

### 偏度

在许多回归算法中，尤其是线性模型，常常假设数值型特征服从正态分布。我们先来计算一下各个数值特征的偏度：


```python
# Check the skew of all numerical features
skewness = df_imputed.select([fn.skewness(var).alias(var) for var in numerical_cols])
skewness = pd.Series(skewness.first().asDict()).sort_values()
print(skewness.head(10))
print(skewness.tail(10))
```

    FLAG_MOBIL                     -554.534039
    FLAG_CONT_MOBILE                -23.081060
    YEARS_BEGINEXPLUATATION_MEDI    -21.825280
    YEARS_BEGINEXPLUATATION_AVG     -21.744660
    YEARS_BEGINEXPLUATATION_MODE    -20.686068
    DAYS_EMPLOYED                    -2.295700
    YEARS_BUILD_MODE                 -1.889130
    YEARS_BUILD_MEDI                 -1.747004
    YEARS_BUILD_AVG                  -1.744856
    FLAG_EMP_PHONE                   -1.664878
    dtype: float64
    FLAG_DOCUMENT_20              44.364680
    FLAG_DOCUMENT_21              54.612673
    FLAG_DOCUMENT_17              61.213842
    FLAG_DOCUMENT_7               72.173756
    FLAG_DOCUMENT_4              110.893823
    AMT_REQ_CREDIT_BUREAU_QRT    141.400225
    FLAG_DOCUMENT_2              153.791067
    FLAG_DOCUMENT_10             209.588031
    AMT_INCOME_TOTAL             391.557744
    FLAG_DOCUMENT_12             392.112866
    dtype: float64


 可以看到这些特征的偏度较高，因此我们尝试变换，让数据接近正态分布。

### QQ图

以AMT_CREDIT特征为例，我们画出分布图和QQ图。

> Quantile-Quantile图是一种常用的统计图形，用来比较两个数据集之间的分布。它是由标准正态分布的分位数为横坐标，样本值为纵坐标的散点图。如果QQ图上的点在一条直线附近，则说明数据近似于正态分布，且该直线的斜率为标准差，截距为均值。


```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot, norm

def norm_comparison_plot(series):
    series = pd.Series(series)
    mu, sigma = norm.fit(series)
    kurt, skew = series.kurt(), series.skew()
    print(f"Kurtosis: {kurt:.2f}", f"Skewness: {skew:.2f}", sep='\t')
    
    fig = plt.figure(figsize=(10, 4))
    # Now plot the distribution
    ax1 = fig.add_subplot(121)
    ax1.set_title('Distribution')
    ax1.set_ylabel('Frequency')
    sns.distplot(series, fit=norm, ax=ax1)
    ax1.legend(['dist','kde','norm'],f'Normal dist. ($\mu=$ {mu:.2f} and $\sigma=$ {sigma:.2f} )', loc='best')
    # Get also the QQ-plot
    ax2 = fig.add_subplot(122)
    probplot(series, plot=plt)
```


```python
sample = df_imputed.select('AMT_CREDIT').sample(0.1).toPandas()
norm_comparison_plot(sample['AMT_CREDIT'])
plt.show()                                                                                 
```

    Kurtosis: 2.06	Skewness: 1.26


![](/img/feature_engineering_on_spark/preproccessing_output_95_2.png)


### 非线性变换

最常用的是log变换。对于含有负数的特征，可以先min-max缩放到[0,1]之间后再做变换。

这里我们对AMT_INCOME_TOTAL特征做log变换


```python
# log-transformation of skewed features.
sample_transformed = df_imputed.select(fn.ln('AMT_CREDIT')).sample(0.1).toPandas()
norm_comparison_plot(sample_transformed.iloc[:, 0])
plt.show()        
```

    Kurtosis: -0.27	Skewness: -0.33


![](/img/feature_engineering_on_spark/preproccessing_output_97_2.png)


可以看到经过log变换后，基本符合正态分布了。

## Baseline

至此，数据预处理已经基本完毕    


```python
df_prepared = df_imputed
print(f'dataset shape: {df_prepared.count(), len(df_prepared.columns)}')
print(pd.Series(dict(df_prepared.dtypes)).value_counts())
```

    dataset shape: (307511, 122)
    double    65
    int       45
    vector    10
    float      2
    Name: count, dtype: int64

规范特征名


```python
new_colnames = {c: c.replace('/','or').replace(' ','_').replace(',','_or') for c in df_prepared.columns}
df_prepared = df_prepared.withColumnsRenamed(new_colnames)
```

### 交叉验证

我们可以选择模型开始训练了。我们准备选择XGBoost模型训练结果作为baseline。   

spark内置的交叉验证CrossValidator主要用于超参数调优，我们重新定义一个交叉验证函数。


```python
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
```


```python
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
        n_estimators=1200,
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
features = df_prepared.drop('SK_ID_CURR', 'label').columns
feature_importances = score_dataset(df_prepared, inputCols=features)
```


    [0] train's areaUnderROC: 0.8817445932752518,  valid's areaUnderROC: 0.7567778599507636
    [1] train's areaUnderROC: 0.8858137153416724,  valid's areaUnderROC: 0.754088602137405
    [2] train's areaUnderROC: 0.8830645318540977,  valid's areaUnderROC: 0.755218312522418
    cv_agg's valid auc: 0.7554 +/- 0.00110


```python
df_prepared.write.bucketBy(100, "SK_ID_CURR").mode("overwrite").saveAsTable("home_credit_default_risk.prepared_data")   
```

### 特征重要性


```python
feature_importances['fscore'].head(15)
```


    NONLIVINGAPARTMENTS_MEDI        4420.333333
    NONLIVINGAREA_MEDI              4300.666667
    YEARS_BEGINEXPLUATATION_MODE    4240.000000
    COMMONAREA_MODE                 4098.666667
    ELEVATORS_MODE                  4023.666667
    NONLIVINGAPARTMENTS_AVG         3947.000000
    LIVINGAREA_AVG                  3862.666667
    YEARS_BUILD_MODE                3781.000000
    NONLIVINGAREA_AVG               3455.333333
    LIVINGAREA_MEDI                 3313.666667
    BASEMENTAREA_MODE               3160.666667
    LIVINGAPARTMENTS_AVG            2819.333333
    LIVINGAPARTMENTS_MEDI           2635.000000
    YEARS_BUILD_MEDI                2312.666667
    ENTRANCES_MODE                  1947.666667
    Name: fscore, dtype: float64


```python
spark.stop()
```
