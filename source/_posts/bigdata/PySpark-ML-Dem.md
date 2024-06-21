---
title: PySpark机器学习Demo
tags:
  - Python
  - Spark
  - 机器学习
categories:
  - Big Data
  - Spark
cover: /img/spark-install.jpg
top_img: /img/apache-spark-top-img.svg
abbrlink: 90489eb7
date: 2024-05-17 21:50:00
description: spark 机器学习库
---

Jupyter Notebook 代码连接：[machine_learning_on_spark_demo](/ipynb/machine_learning_on_spark_demo)

# Step 1: Import required libraries and initialize SparkSession


```python
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
import pyspark.sql.functions as fn
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from xgboost.spark import SparkXGBClassifier
import xgboost as xgb

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

    24/05/17 21:38:51 WARN SparkSession: Using an existing Spark session; only runtime SQL configurations will take effect.


# Step 2: Load the dataset


```python
path = '/Users/***/Documents/Project/datasets/Home-Credit-Default-Risk/prepared_data.csv'
data = spark.read.format("csv").options(inferSchema="true", header=True).load(f"file:///{path}")
```


```python
columns, dtypes = zip(*data.dtypes)
pd.Series(dtypes).value_counts()
```


    double    147
    int        12
    Name: count, dtype: int64


```python
print(f"dataset shape: (data.count(), len(columns))")
```

    dataset shape: (data.count(), len(columns))

```python
data = data.withColumnRenamed("TARGET", "label")

# Check if the data is unbalanced
data.groupBy("label").count().show()
```

    +-----+------+
    |label| count|
    +-----+------+
    |    1| 24825|
    |    0|282686|
    +-----+------+

# Step 3: Data preprocessing


```python
# Convert the categorical labels in the target column to numerical values (optional)
indexer = StringIndexer(
    inputCol="TARGET", 
    outputCol="label"
)
```

StringIndexer 索引的范围从 0 开始，索引构建的顺序为字符标签的频率，优先编码频率较大的标签，所以出现频率最高的标签为 0。


```python
# Assemble the feature columns into a single vector column
feature_names = data.drop("SK_ID_CURR", "label").columns
assembler = VectorAssembler(
    inputCols=feature_names,
    outputCol="features"
)
```


```python
# Split the data into train and test.
train, test = data.randomSplit([0.75, 0.25], seed=SEED)
```

# Step 4: Hyperparameter Tuning & Building a model (optional)

```python
# Create an Estimator.
classifier = SparkXGBClassifier(
    label_col="label", 
    features_col="features",
    eval_metric='auc',
    scale_pos_weight=11,
    verbosity=0
)

# Assemble all the steps into a pipeline.
pipeline = Pipeline(stages=[assembler, classifier])
```

```python
# We use a ParamGridBuilder to construct a grid of parameters to search over.
# CrossValidator will try all combinations of values and determine best model using
# the evaluator.
paramGrid = ParamGridBuilder() \
    .addGrid(classifier.learning_rate, [0.01]) \
    .addGrid(classifier.max_depth, [4, 6, 8]) \
    .addGrid(classifier.subsample, [1.0]) \
    .addGrid(classifier.colsample_bytree, [0.33, 0.66]) \
    .addGrid(classifier.reg_alpha, [0.5, 5.0, 50]) \
    .addGrid(classifier.reg_lambda, [15]) \
    .addGrid(classifier.n_estimators, [500, 1000, 1500]) \
    .build()


evaluator = BinaryClassificationEvaluator(labelCol="label", metricName='areaUnderROC')

# A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
crossval = CrossValidator(
    estimator=pipeline, 
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=4
)
```

```python
# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(train)
```

```python
bestModel = cvModel.bestModel
xgbModel = bestModel.stages[1]
```

```python
print("Best value: ", max(cvModel.avgMetrics))
print("Best params: ")
params = ['learning_rate', 'max_depth', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda', 'n_estimators']
params = {key: xgbModel.getOrDefault(key) for key in params}
for key, value in params.items():
    print(f"    {key}: {value}")
```

# Step 5: Training


```python
# Create an Estimator.
classifier = SparkXGBClassifier(
    features_col='features', 
    label_col='label',
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

# Assemble all the steps into a pipeline.
pipeline = Pipeline(stages=[assembler, classifier])

# Fit the pipeline to training dataset.
model = pipeline.fit(train)
xgbModel = model.stages[1]
```

    Java HotSpot(TM) 64-Bit Server VM warning: CodeCache is full. Compiler has been disabled.
    Java HotSpot(TM) 64-Bit Server VM warning: Try increasing the code cache size using -XX:ReservedCodeCacheSize=
    [Stage 5:=======>                                                   (1 + 7) / 8]
    
    CodeCache: size=131072Kb used=31129Kb max_used=32163Kb free=99942Kb
     bounds [0x0000000102608000, 0x00000001045c8000, 0x000000010a608000]
     total_blobs=10963 nmethods=9953 adapters=920
     compilation: disabled (not enough contiguous free space left)


    2024-05-17 21:38:59,947 INFO XGBoost-PySpark: _fit Running xgboost-2.0.3 on 1 workers with
    	booster params: {'objective': 'binary:logistic', 'colsample_bytree': 0.35, 'device': 'cpu', 'learning_rate': 0.015, 'max_depth': 8, 'reg_alpha': 65, 'reg_lambda': 15, 'scale_pos_weight': 11, 'subsample': 1.0, 'verbosity': 0, 'eval_metric': 'auc', 'nthread': 1}
    	train_call_kwargs_params: {'verbose_eval': True, 'num_boost_round': 1200}
    	dmatrix_kwargs: {'nthread': 1, 'missing': nan}
    [21:39:06] task 0 got new rank 0                                    (0 + 1) / 1]
    2024-05-17 21:40:20,857 INFO XGBoost-PySpark: _fit Finished xgboost training!   


# Step 6: Evaluate the model performance

这是一个二分类问题，先定义一个提取正样本得分的函数


```python
def extractProbability(row):
    d = row.asDict()
    d['probability'] = float(d['probability'][1])
    d['label'] =  float(d.get('label', 0.0))
    return Row(**d)
```

由于PySpark 中并没有在评估模块中直接提供ROC曲线的提取方式，因此我们还要定义一个类去提取：


```python
from pyspark.mllib.evaluation import BinaryClassificationMetrics

class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)
    
    def _to_list(self, rdd):
        points = []
        for row in rdd.collect():
            points += [(float(row._1()), float(row._2()))]
        return points
    
    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        points = self._to_list(rdd)
        return zip(*points)  # return tuple(fpr, tpr)
```

模型得分


```python
def binary_classification_report(model, data):
    predictions = model.transform(data).select('rawPrediction', 'probability', 'label')
    from pyspark.storagelevel import StorageLevel
    predictions.persist(StorageLevel.MEMORY_AND_DISK)
    evaluator = BinaryClassificationEvaluator()
    auc = evaluator.setMetricName('areaUnderROC').evaluate(predictions)
    prob = predictions.select('label', 'probability').rdd.map(extractProbability)
    fpr, tpr = CurveMetrics(prob).get_curve('roc')
    tn = predictions.filter("label == 0 and prediction == 0").count()
    fp = predictions.filter("label == 0 and prediction == 1").count()
    fn = predictions.filter("label == 1 and prediction == 0").count()
    tp = predictions.filter("label == 1 and prediction == 1").count()
    predictions.unpersist()
    return {
        'fpr': fpr,
        'tpr': tpr, 
        'ks': np.max(np.array(tpr) - np.array(fpr)),
        'auc': auc,
        'confusionMatrix': np.array([[tn, fp], [fn, tp]]),
        'accuracy': (tp + tn) / (tn + fp + fn + tp),
        'precision': tp / (fp + tp),
        'recall': tp / (fn + tp)
    }

# the model performance
train_report = binary_classification_report(model, train)
test_report = binary_classification_report(model, test)
for label, stats in [('train data', train_report), ('test data', test_report)]:
    print(label, end=":\n  ")
    print(
        f"auc: {stats['auc']:.5f}", 
        f"accuracy: {stats['accuracy']:.5f}", 
        f"recall: {stats['recall']:.5f}", 
        sep = '\n  '
    )
```


    train data:
      auc: 0.85391
      accuracy: 0.77220
      recall: 0.77017
    test data:
      auc: 0.76201
      accuracy: 0.75205
      recall: 0.62115


ROC 曲线


```python
# Plot ROC curve
def plot_roc_curve(fprs, tprs, aucs, labels):
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # Plotting ROC and computing AUC scores
    for fpr, tpr, auc, label in zip(fprs, tprs, aucs, labels):
    	plt.plot(fpr, tpr, label = f"{label} ROC(auc={auc:.4f})")
    plt.legend(loc = 'lower right')

# plot_roc_curve(
#     fprs = (train_report['fpr'], test_report['fpr']),
#     tprs = (train_report['tpr'], test_report['tpr']),
#     aucs = (train_report['auc'], test_report['auc']),
#     labels = ('train', 'valid')
# )
```

# Step 7: Show feature importance


```python
feature_imp = xgbModel.get_feature_importances()
indices = [int(name[1:]) for name in feature_imp.keys()]

feature_imp = pd.Series(
    feature_imp.values(),
    index=np.array(feature_names)[indices]
).sort_values(ascending=False)

print(feature_imp.head(20))
```

    DAYS_BIRTH                      5783.0
    AMT_ANNUITY                     5666.0
    DAYS_REGISTRATION               5428.0
    DAYS_ID_PUBLISH                 5412.0
    EXT_SOURCE_2                    5188.0
    EXT_SOURCE_3                    5103.0
    AMT_CREDIT                      4899.0
    DAYS_LAST_PHONE_CHANGE          4783.0
    DAYS_EMPLOYED                   4678.0
    anomaly_score                   4576.0
    EXT_SOURCE_1                    4532.0
    AMT_GOODS_PRICE                 4360.0
    REGION_POPULATION_RELATIVE      3965.0
    AMT_INCOME_TOTAL                3944.0
    ORGANIZATION_TYPE               3164.0
    OWN_CAR_AGE                     2975.0
    HOUR_APPR_PROCESS_START         2768.0
    OCCUPATION_TYPE                 2461.0
    AMT_REQ_CREDIT_BUREAU_YEAR      1831.0
    YEARS_BEGINEXPLUATATION_MODE    1658.0
    dtype: float64


# Step 8: Visualize the model


```python
bst = xgbModel.get_booster()
# xgb.plot_tree(bst, num_trees=1000)
```

# Step 9: Save and load the model 


```python
# Save model
# model.save("xgbModel")
```


```python
# Load the model
# from pyspark.ml.classification import SparkXGBClassifierModel
# loaded_model = SparkXGBClassifierModel.load("xgbModel")
```

# Step 10: Predict


```python
# Make predictions on test data. 
predictions = model.transform(test)
selected = predictions.rdd.map(extractProbability).toDF().select("SK_ID_CURR", "probability", "prediction")

# Select example rows to display.
selected.show(5)
```

    2024-05-17 21:41:02,957 INFO XGBoost-PySpark: predict_udf Do the inference on the CPUs
    INFO:XGBoost-PySpark:Do the inference on the CPUs                   (0 + 1) / 1]


    +----------+-------------------+----------+
    |SK_ID_CURR|        probability|prediction|
    +----------+-------------------+----------+
    |    100004|0.18962906301021576|       0.0|
    |    100009|0.08838339149951935|       0.0|
    |    100011| 0.4154616594314575|       0.0|
    |    100012| 0.1840885430574417|       0.0|
    |    100017|0.23958267271518707|       0.0|
    +----------+-------------------+----------+
    only showing top 5 rows


```python
# save predictions
# selected.write.mode('overwrite').saveAsTable('predictions')
```

# Step 11: Clean up


```python
spark.stop()
```

# Appendices: ROC curve

PySpark 中并没有在评估模块中直接提供ROC曲线的提取方式，我们可以从以下三个渠道获得：
- MLlib (DataFrame-based)中的部分分类模型，可以通过summary属性/方法获得
- Spark 在Scala API中提供了提取ROC曲线的方式，因此我们需要从Scala模块中借用
- 自定义ROC提取函数

支持summary属性/方法的模型如下：

```python
# Returns the ROC curve, which is a Dataframe having two fields (FPR, TPR) 
# with (0.0, 0.0) prepended and (1.0, 1.0) appended to it.

roc_df = LinearSVCModel.summary().roc
roc_df = LogisticRegressionModel.summary.roc
roc_df = RandomForestClassificationModel.summary.roc
roc_df = MultilayerPerceptronClassificationModel.summary().roc
roc_df = FMClassificationModel.summary().roc
```

scala中的提取代码如下：

```scala
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

// Instantiate metrics object
val metrics = new BinaryClassificationMetrics(predictionAndLabels)

// Precision-Recall Curve
val PRC = metrics.pr

// AUPRC
val auPRC = metrics.areaUnderPR
println(s"Area under precision-recall curve = $auPRC")

// ROC Curve
val roc = metrics.roc

// AUROC
val auROC = metrics.areaUnderROC
println(s"Area under ROC = $auROC")
```

在 PySpark 中定义一个子类借用scala接口：

```python
from pyspark.mllib.evaluation import BinaryClassificationMetrics

class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)
    
    def _to_list(self, rdd):
        points = []
        # Note this collect could be inefficient for large datasets
        # considering there may be one probability per datapoint (at most)
        # The Scala version takes a numBins parameter,
        # but it doesn't seem possible to pass this from Python to Java
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2,
            # which doesn't appear to have a py4j mapping
            points += [(float(row._1()), float(row._2()))]
        return points
    
    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        points = self._to_list(rdd)
        return zip(*points)  # return tuple(fpr, tpr)
```

定义子类后具体使用如下：

```python
from pyspark.sql import Row
def extractProbability(row, labelCol='label', probabilityCol='probability'):
    return Row(label = float(row[labelCol]), probability = float(row['probability'][1]))

pred_df = predictions.rdd.map(extractProbability)
fpr, tpr = CurveMetrics(pred_df).get_curve('roc')
```

参考sklearn自定义函数提取ROC：

```python
from pyspark.sql import Window, functions as fn
from pyspark.sql import feature as ft

def roc_curve_on_spark(predictions, labelCol='label', probabilityCol='probability'):
    """
    Returns the receiver operating characteristic (ROC) curve,
        which is a Dataframe having two fields (FPR, TPR) with
        (0.0, 0.0) prepended and (1.0, 1.0) appended to it.
    """
    
    roc = predictions.select(labelCol, probabilityCol)
    
    # window functions
    window = Window.orderBy(fn.desc(probabilityCol))
    
    # accumulate the true positives with decreasing threshold
    roc = roc.withColumn('tps', fn.sum(roc[labelCol]).over(window))
    # accumulate the false positives with decreasing threshold
    roc = roc.withColumn('fps', fn.sum(fn.lit(1) - roc[labelCol]).over(window))

    # The total number of negative samples
    numPositive = roc.tail(1)[0]['tps']
    numNegative = roc.tail(1)[0]['fps']

    roc = roc.withColumn('tpr', roc['tps'] / fn.lit(numPositive))
    roc = roc.withColumn('fpr', roc['fps'] / fn.lit(numNegative))
    
    roc = roc.dropDuplicates(subset=[probabilityCol]).select('fpr', 'tpr')

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    start_row = spark.createDataFrame([(0.0, 0.0)], schema=roc.schema)
    roc = start_row.unionAll(roc)
    
    return roc
```
