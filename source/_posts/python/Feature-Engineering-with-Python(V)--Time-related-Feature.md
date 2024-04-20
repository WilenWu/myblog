---
title: 特征工程(V)--时间序列特征
tags:
  - Python
categories:
  - Python
  - 'Machine Learning'
cover: /img/FeatureEngine.png
top_img: /img/sklearn-top-img.svg
description: 
abbrlink: 76296591
date: 2024-04-18 23:40:52
---

有这么一句话在业界广泛流传：数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。由此可见，特征工程在机器学习中占有相当重要的地位。在实际应用当中，可以说特征工程是机器学习成功的关键。

特征工程是数据分析中最耗时间和精力的一部分工作，它不像算法和模型那样是确定的步骤，更多是工程上的经验和权衡。因此没有统一的方法。这里只是对一些常用的方法做一个总结。

特征工程包含了 Data PreProcessing（数据预处理）、Feature Extraction（特征提取）、Feature Selection（特征选择）和 Feature construction（特征构造）等子问题。

Jupyter Notebook 代码连接：[feature_engineering_demo_p5_time_series](/ipynb/feature_engineering_demo_p5_time_series)

导入必要的包


```python
import numpy as np
import pandas as pd
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc

# Setting configuration.
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

SEED = 42
```

# 时间序列特征

本案例并不涉及时序特征（time series），但在其他许多场景，时序特征处理可能会非常复杂。时序特征的处理方法有很多，这里参考sklearn中关于时间特征的案例简单介绍几种有代表性的方法。这些方法可以帮助我们更好地理解和利用时间序列数据中的模式和趋势，从而提高机器学习模型的性能和预测准确度。

参考资料：   
[Time-related feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)    
[Lagged features for time series forecasting](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html)

## 数据探索

Data exploration


```python
# We start by loading the data from the OpenML repository.
from sklearn.datasets import fetch_openml

bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True, parser="pandas")
df = bike_sharing.frame
```


```python
df.head()
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
      <th>season</th>
      <th>year</th>
      <th>month</th>
      <th>hour</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>feel_temp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>spring</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>False</td>
      <td>6</td>
      <td>False</td>
      <td>clear</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>0.81</td>
      <td>0.0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>spring</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>6</td>
      <td>False</td>
      <td>clear</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spring</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>False</td>
      <td>6</td>
      <td>False</td>
      <td>clear</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>spring</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>False</td>
      <td>6</td>
      <td>False</td>
      <td>clear</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>spring</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>False</td>
      <td>6</td>
      <td>False</td>
      <td>clear</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


```python
X = df.drop('count', axis=1)
y = df['count']
```

目标变量为共享单车的每小时租赁次数


```python
# Get a quick understanding of the periodic patterns of the data

fig, ax = plt.subplots(figsize=(12, 4))
average_week_demand = df.groupby(["weekday", "hour"])["count"].mean()
average_week_demand.plot(ax=ax)
_ = ax.set(
    title="Average hourly bike demand during the week",
    xticks=[i * 24 for i in range(7)],
    xticklabels=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    xlabel="Time of the week",
    ylabel="Number of bike rentals",
)
```


![](/img/feature_engineering_with_python/time_series_output_9_0.png)
    


由于数据集是按时间排序的，我们使用基于时间的交叉验证


```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

ts_cv = TimeSeriesSplit(
    n_splits=3,  # to keep the notebook fast enough on common laptops
    gap=48,  # 2 days data gap between train and test
    max_train_size=10000,  # keep train sets of comparable sizes
    test_size=3000,  # for 2 or 3 digits of precision in scores
)
all_splits = list(ts_cv.split(X, y))

train_idx, test_idx = all_splits[0]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```


```python
all_splits
```


    [(array([   0,    1,    2, ..., 8328, 8329, 8330]),
      array([ 8379,  8380,  8381, ..., 11376, 11377, 11378])),
     (array([ 1331,  1332,  1333, ..., 11328, 11329, 11330]),
      array([11379, 11380, 11381, ..., 14376, 14377, 14378])),
     (array([ 4331,  4332,  4333, ..., 14328, 14329, 14330]),
      array([14379, 14380, 14381, ..., 17376, 17377, 17378]))]

构造评估函数


```python
import numpy as np
from sklearn.model_selection import cross_validate

def evaluate(model, X, y, cv, model_prop=None, model_step=None):
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
        return_estimator=model_prop is not None,
    )
    if model_prop is not None:
        if model_step is not None:
            values = [
                getattr(m[model_step], model_prop) for m in cv_results["estimator"]
            ]
        else:
            values = [getattr(m, model_prop) for m in cv_results["estimator"]]
        print(f"Mean model.{model_prop} = {np.mean(values)}")
    mae = -cv_results["test_neg_mean_absolute_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    print(
        f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
        f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}"
    )
```


```python
from sklearn.ensemble import HistGradientBoostingRegressor

gbrt = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42)
categorical_columns = X.columns[X.dtypes == "category"]
print("Categorical features:", categorical_columns.tolist())

evaluate(gbrt, X, y, cv=ts_cv, model_prop="n_iter_")
```

    Categorical features: ['season', 'holiday', 'workingday', 'weather']
    Mean model.n_iter_ = 100.0
    Mean Absolute Error:     61.872 +/- 17.473
    Root Mean Squared Error: 91.119 +/- 23.725


## 时间组件

一般情况下，时间变量都存储在字符中，我们需要先将字符特征转换为时间戳。由于共享单车数据集中没有时间变量，我们自行构造时间数据演示。


```python
ts_df = pd.DataFrame([
    '2019-01-01 01:22:26', '2019-02-02 04:34:52', '2019-03-03 06:16:40',
    '2019-04-04 08:11:38', '2019-05-05 10:52:39', '2019-06-06 12:06:25',
    '2019-07-07 14:05:25', '2019-08-08 16:51:33', '2019-09-09 18:28:28',
    '2019-10-10 20:55:12', '2019-11-11 22:55:12', '2019-12-12 00:55:12',
], columns=['time'])

ts_df['time'] = pd.to_datetime(ts_df['time'])
print(ts_df.head())
```

                     time
    0 2019-01-01 01:22:26
    1 2019-02-02 04:34:52
    2 2019-03-03 06:16:40
    3 2019-04-04 08:11:38
    4 2019-05-05 10:52:39


然后，根据时间所在的自然周期，将一个时间特征转化为若干个离散特征，这种方法在分析具有明显时间趋势的问题比较好用。


```python
ts_df['year'] = ts_df['time'].dt.year
ts_df['month'] = ts_df['time'].dt.month
ts_df['day'] = ts_df['time'].dt.day
ts_df['hour'] = ts_df['time'].dt.hour
ts_df['minute'] = ts_df['time'].dt.minute
ts_df['second'] = ts_df['time'].dt.second
```

进一步提取


```python
ts_df['season'] = ts_df['time'].dt.quarter
ts_df['season'] = ts_df['season'].map({0: "spring", 1: "summer", 2: "spring", 3: "winter"})

# ts_df['weekofyear'] = ts_df['time'].dt.weekofyear
ts_df['dayofweek'] = ts_df['time'].dt.dayofweek
```

还可以提取小时所在每一天的周期：凌晨，上午，下午，晚上


```python
ts_df['hour_section'] = (ts_df['hour'] // 6).astype(str)
```

布尔特征提取


```python
ts_df['is_leap_year'] = ts_df['time'].dt.is_leap_year  # 是否闰年
ts_df['is_month_start'] = ts_df['time'].dt.is_month_start
ts_df['is_month_end'] = ts_df['time'].dt.is_month_end
ts_df['is_quarter_start'] = ts_df['time'].dt.is_quarter_start
ts_df['is_quarter_end'] = ts_df['time'].dt.is_quarter_end
ts_df['is_year_start'] = ts_df['time'].dt.is_year_start
ts_df['is_year_end'] = ts_df['time'].dt.is_year_end
ts_df['is_weekend'] = ts_df['dayofweek'].isin([5, 6])
# ts_df['is_holiday'] = ts_df['time'].isin(holiday_list)  # 需要罗列假日时间
```


```python
ts_df.head(5)
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
      <th>time</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
      <th>minute</th>
      <th>second</th>
      <th>season</th>
      <th>dayofweek</th>
      <th>hour_section</th>
      <th>is_leap_year</th>
      <th>is_month_start</th>
      <th>is_month_end</th>
      <th>is_quarter_start</th>
      <th>is_quarter_end</th>
      <th>is_year_start</th>
      <th>is_year_end</th>
      <th>is_weekend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-01 01:22:26</td>
      <td>2019</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>22</td>
      <td>26</td>
      <td>summer</td>
      <td>1</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-02-02 04:34:52</td>
      <td>2019</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>34</td>
      <td>52</td>
      <td>summer</td>
      <td>5</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-03-03 06:16:40</td>
      <td>2019</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>16</td>
      <td>40</td>
      <td>summer</td>
      <td>6</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-04-04 08:11:38</td>
      <td>2019</td>
      <td>4</td>
      <td>4</td>
      <td>8</td>
      <td>11</td>
      <td>38</td>
      <td>spring</td>
      <td>3</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-05-05 10:52:39</td>
      <td>2019</td>
      <td>5</td>
      <td>5</td>
      <td>10</td>
      <td>52</td>
      <td>39</td>
      <td>spring</td>
      <td>6</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



差分编码：对于时间型数据来说，即可以把它转换成离散值，也可以计算出时间变量到某一时间之间的数值差距，从而转化为连续值。


```python
ts_df['diff'] = ts_df['time'] - pd.Timestamp('2019-01-01')
ts_df['diff'].dt.days
ts_df['diff'].dt.seconds

ts_df['diff_hours'] = ts_df['diff'].values.astype("timedelta64[h]").astype('int')  # hours
```


```python
ts_df['diff'].values.astype("timedelta64[h]")
```


    array([   1,  772, 1470, 2240, 2986, 3756, 4502, 5272, 6042, 6788, 7558,
           8280], dtype='timedelta64[h]')

> 注意，这里的days属性是完全忽略时分秒属性，seconds则是忽略了天数的计算结果。后面的diff_hours则是将时间间隔转换成相差的小时数。

## one-hot编码

Time-steps as categories

由于时间特征是离散的，我们可以忽略排序，使用one-hot编码，从而给线性模型带来更大的灵活性。


```python
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
alphas = np.logspace(-6, 6, 25)
one_hot_linear_pipeline = make_pipeline(
    ColumnTransformer(
        transformers=[
            ("categorical", one_hot_encoder, categorical_columns),
            ("one_hot_time", one_hot_encoder, ["hour", "weekday", "month"]),
        ],
        remainder=MinMaxScaler(),
    ),
    RidgeCV(alphas=alphas),
)

evaluate(one_hot_linear_pipeline, X, y, cv=ts_cv)
```

    Mean Absolute Error:     100.844 +/- 6.861
    Root Mean Squared Error: 132.145 +/- 2.966


绘制拟合曲线：


```python
one_hot_linear_pipeline.fit(X_train, y_train)
one_hot_linear_predictions = one_hot_linear_pipeline.predict(X_test)

plt.figure(figsize=(12, 4))
plt.title("Predictions by one-hot linear models")
plt.plot(y_test.values[-96:], "x-", alpha=0.2, label="Actual demand", color="black")
plt.plot(one_hot_linear_predictions[-96:], "x-", label="One-hot time features")
plt.legend()
plt.show()
```


​    
![](/img/feature_engineering_with_python/time_series_output_34_0.png)
​    


## 周期性编码

在我们之前看到的拟合线中，呈现出了阶梯状的特点。这是因为每个虚拟变量都被单独处理，缺乏连续性。然而，时间等变量具有明显的周期性连续性。这时，我们可以尝试对时间特征进行周期编码(cyclical encoding)，重新编码后，周期范围内的第一个值和最后一个值之间还可以保持平滑。

### 三角函数变换

cyclical encoding with trigonometric transformation

我们可以使用正弦/余弦变换进行循环编码。我们可以利用这种变换将周期性的时间特征编码成两个新的特征，从而更好地表达时间的连续性。


```python
from sklearn.preprocessing import FunctionTransformer

def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))
def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))
```

现在，我们可以使用以下策略构建特征提取管道：


```python
cyclic_cossin_transformer = ColumnTransformer(
    transformers=[
        ("categorical", one_hot_encoder, categorical_columns),
        ("month_sin", sin_transformer(12), ["month"]),
        ("month_cos", cos_transformer(12), ["month"]),
        ("weekday_sin", sin_transformer(7), ["weekday"]),
        ("weekday_cos", cos_transformer(7), ["weekday"]),
        ("hour_sin", sin_transformer(24), ["hour"]),
        ("hour_cos", cos_transformer(24), ["hour"]),
    ],
    remainder=MinMaxScaler(),
)
cyclic_cossin_linear_pipeline = make_pipeline(
    cyclic_cossin_transformer,
    RidgeCV(alphas=alphas),
)
evaluate(cyclic_cossin_linear_pipeline, X, y, cv=ts_cv)
```

    Mean Absolute Error:     115.371 +/- 14.248
    Root Mean Squared Error: 159.276 +/- 5.759


绘制拟合曲线


```python
cyclic_cossin_linear_pipeline.fit(X_train, y_train)
cyclic_cossin_linear_predictions = cyclic_cossin_linear_pipeline.predict(X_test)

plt.figure(figsize=(12, 4))
plt.title("Predictions by one-hot linear models")
plt.plot(y_test.values[-96:], "x-", alpha=0.2, label="Actual demand", color="black")
plt.plot(cyclic_cossin_linear_predictions[-96:], "x-", label="One-hot + Trigonometric")
plt.legend()
plt.show()
```


​    
![](/img/feature_engineering_with_python/time_series_output_40_0.png)
​    


上图说明，正弦/余弦特征允许模型拾取主要模式，但不足以完全捕获该系列的动态。

### 周期性样条特征

Periodic spline features

同样我们可以尝试对每个周期性特征使用足够数量的样条变换


```python
from sklearn.preprocessing import SplineTransformer

def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )
```

参数 `extrapolation="periodic"`，使周期范围内的第一个值和最后一个值之间保持平滑。
对于那些有序的离散特征，样条编码在保留更多信息的同时，比one-hot编码更有效。


```python
cyclic_spline_transformer = ColumnTransformer(
    transformers=[
        ("categorical", one_hot_encoder, categorical_columns),
        ("cyclic_month", periodic_spline_transformer(12, n_splines=6), ["month"]),
        ("cyclic_weekday", periodic_spline_transformer(7, n_splines=3), ["weekday"]),
        ("cyclic_hour", periodic_spline_transformer(24, n_splines=12), ["hour"]),
    ],
    remainder=MinMaxScaler(),
)
cyclic_spline_linear_pipeline = make_pipeline(
    cyclic_spline_transformer,
    RidgeCV(alphas=alphas),
)
evaluate(cyclic_spline_linear_pipeline, X, y, cv=ts_cv)
```

    Mean Absolute Error:     98.339 +/- 6.321
    Root Mean Squared Error: 132.398 +/- 2.490


绘制拟合曲线


```python
cyclic_spline_linear_pipeline.fit(X_train, y_train)
cyclic_spline_linear_predictions = cyclic_spline_linear_pipeline.predict(X_test)

plt.figure(figsize=(12, 4))
plt.title("Predictions by one-hot linear models")
plt.plot(y_test.values[-96:], "x-", alpha=0.2, label="Actual demand", color="black")
plt.plot(cyclic_spline_linear_predictions[-96:], "x-", label="One-hot + Spline")
plt.legend()
plt.show()
```


​    
![](/img/feature_engineering_with_python/time_series_output_47_0.png)
​    


## 特征交互

feature interactions

线性模型不能自动捕获特征间的交互效应，因此我们可以显式的引入交互项，然后与之前已经预处理的特征合并。


```python
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PolynomialFeatures

hour_workday_interaction = make_pipeline(
    ColumnTransformer(
        [
            ("cyclic_hour", periodic_spline_transformer(24, n_splines=8), ["hour"]),
            ("workingday", FunctionTransformer(lambda x: x == "True"), ["workingday"]),
        ]
    ),
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
)

# Those features are then combined with the ones already computed in the previous spline-base pipeline. 
cyclic_spline_interactions_pipeline = make_pipeline(
    FeatureUnion(
        [
            ("marginal", cyclic_spline_transformer),
            ("interactions", hour_workday_interaction),
        ]
    ),
    RidgeCV(alphas=alphas),
)
evaluate(cyclic_spline_interactions_pipeline, X, y, cv=ts_cv)
```

    Mean Absolute Error:     84.792 +/- 6.839
    Root Mean Squared Error: 110.399 +/- 7.442

```python
cyclic_spline_interactions_pipeline.fit(X_train, y_train)
cyclic_spline_interactions_predictions = cyclic_spline_interactions_pipeline.predict(X_test)
```


```python
plt.figure(figsize=(12, 4))
plt.title("Predictions by linear models")
plt.plot(y_test.values[-96:], "x-", alpha=0.2, label="Actual demand", color="black")
plt.plot(one_hot_linear_predictions[-96:], "x-", label="One-hot time features")
plt.plot(cyclic_cossin_linear_predictions[-96:], "x-", label="One-hot + Trigonometric")
plt.plot(cyclic_spline_linear_predictions[-96:], "x-", label="One-hot + Spline")
plt.plot(cyclic_spline_interactions_predictions[-96:], "x-", label="Splines + polynomial")
plt.legend()
plt.show()
```


​    
![](/img/feature_engineering_with_python/time_series_output_51_0.png)
​    


## 窗口特征

将时间序列在时间轴上划分窗口是一个常用且有效的方法，包括**滑动窗口**（ 根据指定的单位长度来框住时间序列，每次滑动一个单位），与**滚动窗口**（根据指定的单位长度来框住时间序列，每次滑动窗口长度的多个单位）。窗口分析对平滑噪声或粗糙的数据非常有用，比如移动平均法等，这种方式结合基础的统计方法，即按照时间的顺序对每一个时间段的数据进行统计，从而可以得到每个时间段内目标所体现的特征，进而从连续的时间片段中，通过对同一特征在不同时间维度下的分析，得到数据整体的变化趋势。


```python
count = df["count"]
lagged_df = pd.concat(
    [
        count,
        count.shift(1).rename("lagged_count_1h"),
        count.shift(2).rename("lagged_count_2h"),
        count.shift(3).rename("lagged_count_3h"),
        count.shift(24).rename("lagged_count_1d"),
        count.shift(24 + 1).rename("lagged_count_1d_1h"),
        count.shift(7 * 24).rename("lagged_count_7d"),
        count.shift(7 * 24 + 1).rename("lagged_count_7d_1h"),
        count.shift(1).rolling(24).mean().rename("lagged_mean_24h"),
        count.shift(1).rolling(24).max().rename("lagged_max_24h"),
        count.shift(1).rolling(24).min().rename("lagged_min_24h"),
        count.shift(1).rolling(7 * 24).mean().rename("lagged_mean_7d"),
        count.shift(1).rolling(7 * 24).max().rename("lagged_max_7d"),
        count.shift(1).rolling(7 * 24).min().rename("lagged_min_7d"),
    ],
    axis="columns",
)
lagged_df.tail(10)
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
      <th>count</th>
      <th>lagged_count_1h</th>
      <th>lagged_count_2h</th>
      <th>lagged_count_3h</th>
      <th>lagged_count_1d</th>
      <th>lagged_count_1d_1h</th>
      <th>lagged_count_7d</th>
      <th>lagged_count_7d_1h</th>
      <th>lagged_mean_24h</th>
      <th>lagged_max_24h</th>
      <th>lagged_min_24h</th>
      <th>lagged_mean_7d</th>
      <th>lagged_max_7d</th>
      <th>lagged_min_7d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17369</th>
      <td>247</td>
      <td>203.0</td>
      <td>224.0</td>
      <td>157.0</td>
      <td>160.0</td>
      <td>169.0</td>
      <td>70.0</td>
      <td>135.0</td>
      <td>93.500000</td>
      <td>224.0</td>
      <td>1.0</td>
      <td>67.732143</td>
      <td>271.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17370</th>
      <td>315</td>
      <td>247.0</td>
      <td>203.0</td>
      <td>224.0</td>
      <td>138.0</td>
      <td>160.0</td>
      <td>46.0</td>
      <td>70.0</td>
      <td>97.125000</td>
      <td>247.0</td>
      <td>1.0</td>
      <td>68.785714</td>
      <td>271.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17371</th>
      <td>214</td>
      <td>315.0</td>
      <td>247.0</td>
      <td>203.0</td>
      <td>133.0</td>
      <td>138.0</td>
      <td>33.0</td>
      <td>46.0</td>
      <td>104.500000</td>
      <td>315.0</td>
      <td>1.0</td>
      <td>70.386905</td>
      <td>315.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17372</th>
      <td>164</td>
      <td>214.0</td>
      <td>315.0</td>
      <td>247.0</td>
      <td>123.0</td>
      <td>133.0</td>
      <td>33.0</td>
      <td>33.0</td>
      <td>107.875000</td>
      <td>315.0</td>
      <td>1.0</td>
      <td>71.464286</td>
      <td>315.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17373</th>
      <td>122</td>
      <td>164.0</td>
      <td>214.0</td>
      <td>315.0</td>
      <td>125.0</td>
      <td>123.0</td>
      <td>26.0</td>
      <td>33.0</td>
      <td>109.583333</td>
      <td>315.0</td>
      <td>1.0</td>
      <td>72.244048</td>
      <td>315.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17374</th>
      <td>119</td>
      <td>122.0</td>
      <td>164.0</td>
      <td>214.0</td>
      <td>102.0</td>
      <td>125.0</td>
      <td>26.0</td>
      <td>26.0</td>
      <td>109.458333</td>
      <td>315.0</td>
      <td>1.0</td>
      <td>72.815476</td>
      <td>315.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17375</th>
      <td>89</td>
      <td>119.0</td>
      <td>122.0</td>
      <td>164.0</td>
      <td>72.0</td>
      <td>102.0</td>
      <td>18.0</td>
      <td>26.0</td>
      <td>110.166667</td>
      <td>315.0</td>
      <td>1.0</td>
      <td>73.369048</td>
      <td>315.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17376</th>
      <td>90</td>
      <td>89.0</td>
      <td>119.0</td>
      <td>122.0</td>
      <td>47.0</td>
      <td>72.0</td>
      <td>23.0</td>
      <td>18.0</td>
      <td>110.875000</td>
      <td>315.0</td>
      <td>1.0</td>
      <td>73.791667</td>
      <td>315.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17377</th>
      <td>61</td>
      <td>90.0</td>
      <td>89.0</td>
      <td>119.0</td>
      <td>36.0</td>
      <td>47.0</td>
      <td>22.0</td>
      <td>23.0</td>
      <td>112.666667</td>
      <td>315.0</td>
      <td>1.0</td>
      <td>74.190476</td>
      <td>315.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>17378</th>
      <td>49</td>
      <td>61.0</td>
      <td>90.0</td>
      <td>89.0</td>
      <td>49.0</td>
      <td>36.0</td>
      <td>12.0</td>
      <td>22.0</td>
      <td>113.708333</td>
      <td>315.0</td>
      <td>1.0</td>
      <td>74.422619</td>
      <td>315.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



目标变量的自相关性


```python
df['count'].autocorr(1)
```


    0.8438107986135388

