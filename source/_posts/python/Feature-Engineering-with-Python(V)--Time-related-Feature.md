---
title: 特征工程(I)--时序特征工程
tags:
  - Python
categories:
  - Python
  - 'Machine Learning'
cover: /img/FeatureEngine.png
top_img: /img/sklearn-top-img.svg
abbrlink: 
description: 
date: 
---

# 时序特征工程

本案例并不涉及时序特征（time series），但在其他许多场景，时序特征处理可能会非常复杂。时序特征的处理方法有很多，这里参考sklearn中关于时间特征的案例简单介绍几种有代表性的方法。

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

一般情况下，时间变量都存储在字符中。我们先将字符特征转换为时间格式

```python
df['time'] = pd.to_datetime(df['time'])
```

目标变量为共享单车的每小时租赁次数

```python
# Get a quick understanding of the periodic patterns of the data
import matplotlib.pyplot as plt

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
<img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_cyclical_feature_engineering_001.png"  />

由于数据集是按时间排序的，我们使用基于时间的交叉验证

```python
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

cv_mape_scores = -cross_val_score(
    model, X, y, cv=ts_cv, scoring="neg_mean_absolute_percentage_error"
)
```
评估函数

```python
import numpy as np

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

evaluate(gbrt, X, y, cv=ts_cv, model_prop="n_iter_")
```


## 时间组件提取

首先根据时间所在的自然周期，将一个时间特征转化为若干个离散特征，这种方法在分析具有明显时间趋势的问题比较好用。

```python
df['year'] = df['time'].dt.year
```
自然周期提取：
season: spring
year	
month	
hour	
holiday	
weekday	
workday

season
fall      4496
summer    4409
spring    4242
winter    4232

dt.quarter 季度 
dt.weekofyear 当年的第几周
dt.dayofweek 周几
进一步创建小时所在每一天的周期：凌晨，上午，下午，晚上
hour_section = (df['hour'] // 6).astype('int')



## 差值编码

然后，我们可以计算出时间变量到某一时间之间的数值差距，从而将时间特征转化为连续值。

```python
df['diff'] = df['t1'] - df['t2']
df['diff'].days
df['diff'].seconds

df['diff'].values.astype("timedelta64[h]").astype('int')
```
注意，这里的days属性是完全忽略时分秒属性，seconds则是忽略了天数的计算结果

# 差分

适合处理时间序列变量，一阶差分为$x_n - x_{n-1}$，k阶差分为$x_n - x_{n-k}$。差分的目的是将非平稳序列转换成平稳序列，而平稳序列具备均值回归的特性，更容易预测。但差分的代价是损失信息，阶数越大，损失的就越多

## 类别编码

Time-steps as categories

由于时间特征是离散的，我们可以忽略排序，使用one-hot编码，从而给线性模型带来更大的灵活性。
```python
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

## 三角函数特征

Trigonometric features

我们可以尝试对每个周期性特征进行正弦和余弦变换，重新编码后，还可以让周期范围内的第一个值和最后一个值之间保持平滑。

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

## 周期性样条特征

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
由于使用了参数 `extrapolation="periodic"`，使周期范围内的第一个值和最后一个值之间保持平滑。
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

## 特征交互

feature interactions

 线性模型不能自动捕获特征间的交互效应，因此我们可以显式的引入交互项，然后与之前已经预处理的特征的合并
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

## 时间序列滞后特征
Lagged features for time series forecasting

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