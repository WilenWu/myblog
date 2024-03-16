---
title: 特征工程(II)--数据预处理
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

# 数据预处理

数据预处理是特征工程的最重要的起始步骤，需要把特征预处理成机器学习模型所能接受的形式，我们可以使用sklearn.preproccessing模块来解决大部分数据预处理问题。

本章使用两条线并行处理数据：
- 基于pandas的函数封装实现
- 基于sklearn的pipeline实现

先定义一个计时器，方便后续评估性能。

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
	    result = func(*args, **kwargs)
	    delta = strfdelta(time.time() - click, "{:.0f} hours {:.0f} minutes {:.0f} seconds")
	    print(f"{func.__name__} cost time {delta}")
	    return result
	return wrapper
```

## 数据清洗

数据清洗(Data cleaning)：对数据进行重新审查和校验的过程，目的在于删除重复信息、纠正存在的错误，并提供数据一致性。

首先，根据某个/多个特征值构成的样本ID去重
```python
df.drop_duplicates(subset=[ID_col], keep='last')
```

字符型数字自动转成数字
```python
df = pd.to_numeric(df, errors='ignore')
```

有时，有些数值型特征标识的只是不同类别，其数值的大小并没有实际意义，因此我们将其转化为类别特征。本项目并无此类特征，以 hours_appr_process_start 为示例：
```python
df['hours_appr_process_start '] = df['hours_appr_process_start '].astype("category")
```

接下来，我们根据业务常识，或者使用但不限于箱型图（Box-plot）发现数据中不合理的特征值进行清洗。

注意到，DAYS_BIRTH列（年龄）中的数字是负数，由于它们是相对于当前贷款申请计算的，所以我们将其转化成正数后查看分布
```python
(df['DAYS_BIRTH'] / -365).describe()
```
那些年龄看起来合理，没有异常值。

接下来，我们对其他的 DAYS 特征作同样的分析
```python
days_registration
days_id_publish
own_car_age
app_train['DAYS_EMPLOYED'].describe() 
```
```python
pd.cut(df['DAYS_EMPLOYED'] / -365, bins=10).value_counts()
```
有超过50000个用户的DAYS_EMPLOYED在1000年上，可以猜测这只是缺失值标记。
```python
# Replace the anomalous values with nan
df['DAYS_EMPLOYED'].where(df['DAYS_EMPLOYED']<365243, np.nan, inplace = True)

app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment')
```
可以看到，数据分布基本正常了。

同样，将其他特征的缺失值标记转换成缺失，方便后续统一处理
```python
df = df.replace('XNA', np.nan)
```

最后，使用函数封装以上步骤
```python
df = df.drop_duplicates(subset=[ID_col], keep='last')
X = df.drop(ID_col).copy()
y = X.pop(target)

@timer
def clean(X, y=None):
    X = pd.to_numeric(X), errors='ignore')
    X['hours_appr_process_start '].astype("category", inplace=True)
    X['DAYS_EMPLOYED'].where(X['DAYS_EMPLOYED']<365243, np.nan, inplace=True)
    X.replace('XNA', np.nan, inplace=True)
    X.select_dtypes("object").astype("category", inplace=True)
    return X
```

## 离散特征编码

有很多机器学习算法只能接受数值型特征的输入，不能处理离散值特征，比如线性回归，逻辑回归等线性模型，那么我们需要将离散特征重编码成数值变量。

| 方法         | 函数           | python包              |
| ------------ | -------------- | --------------------- |
| 顺序编码     | OrdinalEncoder | sklearn.preprocessing |
| 顺序编码 | CategoricalDtype | pandas.api.types |
| One-hot 编码 | OneHotEncoder  | sklearn.preprocessing |
| 哑变量变换 | pd.get_dummies | pandas                |
|平均数编码|self-define||

> 实际上，大多数情况下不区分one-hot和dummy。它们的区别是：如果我们的特征有N个取值，one-hot会用N个新的0/1特征代替，dummy只需要N-1个新特征来代替。

不同类型的离散特征有不同的编码方式。
```python
features = df.columns.drop([ID_col, target]).to_list() 
categorical_cols = df.select_dtypes(["object", "category"]).columns.tolist()
numeric_cols = df.select_dtypes("number").columns.tolist()
```

**有序分类特征**实际上表征着潜在的排序关系，我们将这些特征的类别映射成有大小的数字，因此可以用顺序编码。

多数情况下，整型变量都存储了有限的离散值

```python
df.select_dtypes("int32").nunique()
```

```python
# The ordinal (ordered) categorical features
# Pandas calls the categories "levels"

ordered_levels = {
    "NAME_YIELD_GROUP": ["low_action", "low_normal", "middle", "high"],
  	"NAME_EDUCATION_TYPE": ["Lower secondary", "Secondary / secondary special", "Incomplete higher", "Higher education"]
}
```

**无序分类特征**对于树集成模型（tree-ensemble like XGBoost）是可用的，但对于线性模型（like Lasso or Ridge）则必须使用one-hot重编码。

```python
# The nominative (unordered) categorical features
nominal_categories= [col for col in categorical_cols if col not in ordered_levels.keys()]
```

现在我们来看看每个离散变量的类别数

```python
df[nominal_categories].nunique()
```

**使用pandas实现编码**

```python
# Using pandas to encode categorical features
from pandas.api.types import CategoricalDtype

def onehot_encode(X, variables=None, dummy_na=True):
    """
    Replace the categorical variables by the binary variables.
    
    Parameters
    ----------
    X: pd.DataFrame of shape = [n_samples, n_features]
        The data to encode.
        Can be the entire dataframe, not just seleted variables.
    variables: list, default=None
        The list of categorical variables that will be encoded. 
        If None, the encoder will find and encode all variables of type object or categorical by default.
    dummy_na: boolean, default=True

    Returns
    -------
    X_new: pd.DataFrame.
        The encoded dataframe. The shape of the dataframe will be different from
        the original as it includes the dummy variables in place of the of the
        original categorical ones.
    """
    
	# pd.get_dummies automatically convert the categorical column into dummy variables
    if variables is None:
        X = pd.get_dummies(X, dummy_na=True)
        variables = X.select_dtypes(['category', 'object']).columns.to_list()
    else:
        X_dummy = pd.get_dummies(X[variables].astype('category'), dummy_na=True)
        X = pd.concat([X, X_dummy], axis=1, errors='ignore')
        # drop the original non-encoded variables.
        X = X.drop(variables)
    print(f'{len(variables):d} columns were one-hot encoded')
    print(f'Training Features shape: {X.shape}')
	return X

def ordinal_encode(X, levels: dict = None):
    if levels is None:
        variables = X.select_dtypes(['category', 'object']).columns.to_list()
        X[variables].astype("category", copy=False)
    else:
        dtypes = {name: CategoricalDtype(levels[name], ordered=True) for name in levels}
        X.astype(dtypes, copy=False)
        
    # Add a None category for missing values
    # def add_na(x):
    # if "NA" not in x.cat.categories:
    #    return x.cat.add_categories("NA")
    #  else:
    #    return x
    # X = X.apply(add_na)
    
    # The `cat.codes` attribute holds the category levels.
    # For missing values, -1 is the default code.
    X = X.apply(lambda x: x.cat.codes)
	print(f'{len(variables):d} columns were ordinal encoded')
  return X

print(X_train.pipe(onehot_encode, variables=nominal_categories)
             .pipe(ordinal_encode, levels=ordered_levels)
             .head(10))
```


**使用sklearn实现编码**
```python
# Using sklearn
ordinal_encoder = OrdinalEncoder(
	categories=[np.array(levels) for levels in ordered_levels.values()]
	handle_unknown='use_encoded_value', 
	unknown_value=-1,
	encoded_missing_value=-1,
	max_categories=None)

onehot_encoder = OneHotEncoder(
	drop='if_binary', 
	min_frequency=0.02, 
	max_categories=20, 
	handle_unknown='ignore', 
	feature_name_combiner='concat',
	sparse_output=False)

categorical_encoder = ColumnTransformer(
	[('onehot_encoder', onehot_encoder, nominal_categories),
	('ordinal_encoder', ordinal_encoder, ordered_levels.keys().to_list())],
	remainder='passthrough', verbose_feature_names_out=False)

categorical_encoder.fit_transform(X_train)
categorical_encoder.transform(X_valid)
categorical_encoder.get_feature_names_out()
```

**平均数编码**：一般情况下，针对分类特征，我们只需要使用sklearn的OneHotEncoder或OrdinalEncoder进行编码，这类简单的预处理能够满足大多数数据挖掘算法的需求。如果某一个分类特征的可能值非常多（高基数 high cardinality），那么再使用one-hot编码往往会出现维度爆炸。平均数编码（mean encoding）是一种高效的编码方式，在实际应用中，能极大提升模型的性能。

我们可以使用 feature-engine开源包实现平均数编码。[feature-engine](https://feature-engine.trainindata.com/en/latest/)将特征工程中常用的方法进行了封装。

其中变量 OCCUPATION_TYPE （职业类型）和 ORGANIZATION_TYPE类别数较多，准备使用平均数编码
```python
from feature_engine.encoding import MeanEncoder
mean_encoder = MeanEncoder(['OCCUPATION_TYPE', 'ORGANIZATION_TYPE'], missing_values='ignore', ignore_format=True, unseen='ignore')
mean_encoder.fit_transform(X_train, y_train)
mean_encoder.transform(X_valid)
mean_encoder.get_feature_names_out()
```

## 连续特征分箱

Binning Continuous Features

在实际的模型训练过程中，我们也经常对连续特征进行离散化处理，这样能消除特征量纲的影响，同时还能极大减少异常值的影响，增加特征的稳定性。


| 方法     | 函数             | python包              |
| -------- | ---------------- | --------------------- |
| 二值化   | Binarizer        | sklearn.preprocessing |
| 分箱     | KBinsDiscretizer | sklearn.preprocessing |
| 等频分箱 | pd.qcut          | pandas                |
| 等宽分箱 | pd.cut           | pandas                |

分箱主要分为等频分箱、等宽分箱和聚类分箱三种。等频分箱会一定程度受到异常值的影响，而等宽分箱又容易完全忽略异常值信息，从而一定程度上导致信息损失，若要更好的兼顾变量的原始分布，则可以考虑聚类分箱。所谓聚类分箱，指的是先对某连续变量进行聚类（往往是 k-Means 聚类），然后使用样本所属类别。

以年龄对还款的影响为例
```python
# Find the correlation of the positive days since birth and target
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])
```

客户年龄与目标意义呈负相关关系，即随着客户年龄的增长，他们往往会更经常地按时偿还贷款。我们接下来将制作一个核心密度估计图（KDE），直观地观察年龄对目标的影响。 
```python
plt.figure(figsize = (10, 8))
sns.kdeplot(app_train.loc[app_train['DAYS_BIRTH'] / 365, color=app_train['TARGET'])
plt.xlabel('Age (years)')
plt.ylabel('Density')
plt.title('Distribution of Ages');
```

如果我们把年龄分箱：

```python
# Age information into a separate dataframe
age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_groups  = age_data.groupby('YEARS_BINNED').mean()

plt.figure(figsize = (8, 8))
# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
# Plot labeling
plt.xticks(rotation = 75)
plt.xlabel('Age Group (years)')
plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group');
```
有一个明显的趋势：年轻的申请人更有可能不偿还贷款！ 年龄最小的三个年龄组的失败率在10％以上，最老的年龄组为5％。

使用pandas实现分箱
```python
def discretize(X, variables=None, bins=10, strategy="uniform", bucket_labels=None, encoding=None):
    """
    Parameters
    ----------
    bucket_labels: dict, default=None
    """
    if strategy not in ["uniform", "quantile"]:
        raise ValueError("strategy takes only values 'uniform' or 'quantile'")
    if encoding not in [None, "onehot", "ordinal"]:
        raise ValueError("encoding takes only values None, 'onehot' or 'ordinal'")
    
    if strategy == "uniform":
        cut = pd.cut
    elif strategy == "quantile":
        cut = pd.qcut
    
    bucket_labels = {} if bucket_labels is None else bucket_labels
    if variables is None:
        variables = X.select_dtypes("number").columns.to_list()
    X_binned = X[variables].apply(lambda x: cut(labels=bucket_labels.get(x.name)), 
               bins=bins, labels=labels, duplicates='drop')
    if encoding == "onehot":
        X_binned = pd.get_dummies(X_binned, dummy_na=True)
    elif encoding == "ordinal":
        X_binned = X_binned.apply(lambda x: x.cat.codes)

    X = pd.concat(X.drop(variables), X_binned, axis=1)
    return X


discretize(X).head()
```

sklearn.preprocessing 模块中的 KBinsDiscretizer 可以实现等频分箱、等宽分箱或聚类分箱，同时还可以对分箱后的离散特征进一步进行one-hot编码或顺序编码。
```python
from sklearn.preprocessing import KBinsDiscretizer

equal_frequency_discretiser = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
equal_width_discretiser = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
kmeans_cluster_discretiser = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans')

X_binned = equal_width_discretiser.fit_transform(X_train)
X_binned.bin_edges_
```

## 异常值检测

我们在实际项目中拿到的数据往往有不少异常数据，这些异常数据很可能让我们模型有很大的偏差。异常检测的方法有很多，例如3倍标准差、箱线法的单变量标记，或者聚类、iForest和LocalOutlierFactor等无监督学习方法。

|方法| python模块                      |
---| ---------------------------- | 
|分位数检测|self-define|
|3倍标准差原则|self-define|
|聚类检测|self-define|
|One Class SVM| sklearn.svm.OneClassSVM              |  
|Elliptic Envelope| sklearn.linear_model.SGDOneClassSVM  |    
|Elliptic Envelope| sklearn.covariance.EllipticEnvelope  |  
|Isolation Forest| sklearn.ensemble.IsolationForest     |  
|LOF| sklearn.neighbors.LocalOutlierFactor | 

**箱线图检测**根据四分位点判断是否异常。四分位数具有鲁棒性，不受异常值的干扰。通常认为小于 $Q_1-1.5*IQR$ 或大于 $Q_3+1.5*IQR$ 的点为离群点。 

**3倍标准差原则**：假设数据满足正态分布，通常定义偏离均值的 $3\sigma$ 之外内的点为离群点，$\mathbb P(|X-\mu|<3\sigma)=99.73\%$。如果数据不服从正态分布，也可以用远离平均值的多少倍标准差来描述。

使用pandas实现，并封装在transformer中
```python
class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps maximum and/or minimum values of a variable at automatically
    determined values.
    Works only with numerical variables. A list of variables can be indicated. 
    
    Parameters
    ----------
    method: str, 'gaussian' or 'iqr', default='iqr'
        If method='gaussian': 
            - upper limit: mean + 3* std
            - lower limit: mean - 3* std
        If method='iqr': 
            - upper limit: 75th quantile + 3* IQR
            - lower limit: 25th quantile - 3* IQR
            where IQR is the inter-quartile range: 75th quantile - 25th quantile.
    fold: int, default=3   
        You can select how far out to cap the maximum or minimum values.
    """

    def __init__(method='iqr', fold=3, variables=None):
        self.method = method
        self.fold = fold
        self.variables = variables

    def fit(self, X, y=None):
        """
        Learn the values that should be used to replace outliers.

        Parameters
        ----------
        X : pandas dataframe of shape = [n_samples, n_features]
            The training input samples.

        y : pandas Series, default=None
            y is not needed in this transformer. You can pass y or None.
        """

        # check input dataframe
        X = check_X(X)
        
        # Get the names and number of features in the train set.
        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]
        
        # find or check for numerical variables
        numeric_vars = X.select_dtypes("number").columns.to_list()
        if self.variables is None:
            self.variables = numeric_vars
        else:
            self.variables = list(set(numeric_vars) & set(self.variables))

        if self.method == "gaussian":
            mean = X[self.variables].mean()
            bias= [mean, mean]
            scale = X[self.variables].std(ddof=0)
        elif self.method == "iqr":
            Q1, Q3 = X[self.variables].quantile(q=(0.25, 0.75))
            bias = [Q1, Q3]
            scale = Q3 - Q1         
        
        # estimate the end values
        if (scale == 0).any():
            raise ValueError(
                f"Input columns {scale[scale == 0].index.tolist()!r}"
                f" have low variation for method {self.capping_method!r}."
                f" Try other capping methods or drop these columns."
            )
        else:
            self.upper_limit = bias[1] + self.fold * scale
            self.lower_limit = bias[0] - self.fold * scale   

        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]

        return self # always return self!

    def transform(self, X, y=None):
        """
        Cap the variable values.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The data to be transformed.

        Returns
        -------
        X_new: pandas dataframe of shape = [n_samples, n_features]
            The dataframe with the capped variables.
        """

        # check if class was fitted
        check_is_fitted(self)

        # replace outliers
        X[self.variables] = X[self.variables].clip(
            upper=self.upper_limit,
            lower=self.lower_limit
        )
        outiers = (X[self.variables].gt(self.upper_limit) | 
                   X[self.variables].gt(self.lower_limit))
        n = outiers.sum().gt(0).sum()
        print(f"Your selected dataframe has {n} out of {outiers.shape[1]} columns that have outliers.")
        return X

    def get_feature_names_out(input_features=None):
        check_is_fitted(self)

        if input_features is None:
            return self.feature_names_in_
        elif len(input_features) == len(self.n_features_in_):
            # If the input was an array, we let the user enter the variable names.
            return = list(input_features)
        else:
            raise ValueError(
                "The number of input_features does not match the number of "
                "features seen in the dataframe used in fit."
                ) 
```

```python
outlier_capper = OutlierCapper()
_ = outlier_capper.fit_transform(X)
```

sklearn 包目前支持的异常检测算法：
- **One Class SVM**：基于 SVM (使用高斯内核) 的思想在特征空间中训练一个超球面，边界外的点即为异常值。
- **Elliptic Envelope**：假设数据满足正态分布，训练一个椭圆包络线，边界外的点则为离群点 。
- **Isolation Forest**：是一种高效的异常检测算法，它和随机森林类似，但每次分裂特征和划分点（值）时都是随机的，而不是根据信息增益或基尼指数来选择。
-  **LOF**：基于密度的异常检测算法。离群点的局部密度显著低于大部分近邻点，适用于非均匀的数据集。
- **聚类检测**：常用KMeans聚类将训练样本分成若干个簇，如果某一个簇里的样本数很少，而且簇质心和其他所有的簇都很远，那么这个簇里面的样本极有可能是异常特征样本了。

筛选出来的异常样本需要根据实际含义处理：
- 根据异常点的数量和影响，考虑是否将该条记录删除。
- 对数据做 log-scale 变换后消除异常值。
- 通过数据分箱来平滑异常值。
- 使用均值/中位数/众数来修正替代异常点，简单高效。
- 标记异常值或新增异常值得分列。
- 树模型对离群点的鲁棒性较高，可以选择忽略异常值。

我们接下来考虑对数值型变量添加箱线图异常标记，计算iForest得分并标记异常样本。

```python
from sklearn.ensemble import IsolationForest

class CustomIsolationForest(IsolationForest, TransformerMixin):
    """
    Isolation Forest Algorithm.
    Compute the anomaly score of each sample using the IsolationForest algorithm.
    """
    def __init__(self, **kwargs, drop=False):
        super().__init__(**kwargs)
        self.drop = drop
    def transform(self, X, y=None):  
        anomaly_scores = super().decision_function(X)
        pred = super().predict(X)
        n_outiers = pred[pred == -1].size
        if self.drop:
            print(f"Remove {n_outiers} outliers from the dataset")
            return X.loc[pred == 1,:]
        else:
            # Return average anomaly score of X.
            print(f"The number of outiers: {n_outiers}")
            return anomaly_scores.reshape(-1, 1)
    def get_feature_names_out(self, input_features=None):
        if self.drop:
            return self.feature_names_in_
        else:
            return ["anomaly_score"]
```



```python
# fit the model for anomaly detection

```


## 缺失值处理

特征有缺失值是非常常见的，大部分机器学习模型在拟合前需要处理缺失值（Handle Missing Values）。

缺失值统计
```python
# Function to calculate missing values by column
def display_missing(df, threshold=None, verbose=True):
		missing_df = pd.DataFrame({
		"missing_number": df.isna().sum(),  # Total missing values
		"missing_rate": df.isna().mean()   # Proportion of missing values
		}, index=df.columns)
    missing_df = missing_df.query("missing_rate>0").sort_values("missing_rate", ascending=True)
    threshold = 0.25 if threshold is None else threshold
    high_missing = missing_df.query(f"missing_rate>{threshold}")
    # Print some summary information
    if verbose:
      print(f"Your selected dataframe has {missing_df.shape[0]} out of {df.shape[0]} columns that have missing values.")
      print(f"There are {high_missing.shape[0]} columns with more than {threshold:.1%} missing values.")
      print("Columns with high missing rate:", high_missing.index.tolist())
    # Return the dataframe with missing information
    if threshold is None:
      return missing_df
    else:
      return high_missing

# Missing values statistics
print(display_missing(df).head(10))
```

可视化缺失率
```python
plt.xlabel('Percent of missing values', fontsize=15)
plt.ylabel('Features', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
sns.barplot(y="missing_rate", x='index', data=display_missing(df).head(10))
```

缺失值处理通常有两种策略：
- 缺失值可以用常量值估算，也可以使用缺失值所在列的统计信息（平均值、中位数或众数）估算。
- 如果某个特征的缺失值超过阈值（例如20%），那么该特征对模型的贡献就会降低，通常就可以考虑删除该特征。

| 缺失值处理方法  | 函数                   | python包       |
| --------------- | ---------------------- | -------------- |
| 统计量插补      | SimpleImputer          | sklearn.impute |
| 统计量/随机插补 | df.fillna()            | pandas         |
| 多重插补        | IterativeImputer       | sklearn.impute |
| 最近邻插补      | KNNImputer             | sklearn.impute |
| 缺失值删除      | df.dropna()            | pandas         |
| 缺失值标记      | MissingIndicator       | sklearn.impute |
| 缺失值标记      | df.isna(), df.isnull() | pandas         |

首先，删除缺失值超过20%的特征
```python
threshold = int(df.shape[0]*0.8)
df.dropna(axis=1, thresh=threshold)
```

有时，对于每个含有缺失值的列，我们额外添加一列来表示该列中缺失值的位置，在某些应用中，能取得不错的效果。
继续分析之前清洗过的 DAYS_EMPLOYED 异常，我们对缺失数据进行标记，看看他们是否影响客户违约。

```python
app_train.groupby(app_train['DAYS_EMPLOYED'].isna())[target].mean()
```
发现缺失值的逾期率（）低于正常值的逾期率（），与Target的相关性很强，因此新增一列DAYS_EMPLOYED_MISSING 标记。这种处理应该是对线性方法比较有效，而基于树的方法应该可以自动识别。

```python
# Create a flag column
app_train['DAYS_EMPLOYED_MISSING'] = app_train["DAYS_EMPLOYED"].isna()
```

然后，根据业务知识来进行人工填充。

若变量是离散型，且不同值较少，可在编码时转换成哑变量。例如，性别变量 code_gender
```python
pd.get_dummies(df["code_gender"], dummy_na=True)
```

若变量是布尔型，视情况可统一填充为零
```python
df.filter(regex="^FLAG_").fillna(0)
```

如果我们仔细观察一下字段描述，会发现很多缺失值都有迹可循，比如name_type_suite缺失，办理贷款的时候无人陪同，因此可以用 None 来填补。客户的社会关系中有多少30天/60天逾期及申请贷款前1小时/天/周/月/季度/年查询了多少次征信都可填充为数字0。
```python
features_fill_none = ["name_type_suite", "occupation_type", "orgnization_type"]
df[features_fill_none].fillna('None', inplace=True)

features_fill_zero = ["obs_30_cnt_social_circle", "AMT_REQ_CREDIT_BUREAU_HOUR"] 
df[features_fill_zero].fillna(0, inplace=True)
```

对于缺失率较低（小于5%）的数值特征可以用中位数或均值插补，缺失率较低（小于5%）的离散型特征，则可以用众数插补。
```python
features_fill_mean = df.select_dtypes("number").isna().mean().lt(0.05).columns.tolist()
df[features_fill_mean].fillna(df[features_fill_mean].mean(), inplace=True)

features_fill_mode = df[df.select_dtypes("category").isna().mean().lt(0.05)]
df[features_fill_mode].fillna(df[features_fill_mode].mode(), inplace=True)
```

通过之前的相关分析，我们知道LotFrontage这个特征与LotAreaCut和Neighborhood有比较大的关系，所以这里用这两个特征分组后的中位数进行插补，称为条件平均值填充法（Conditional Mean Completer）。
```python
# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
 
# Filling the missing values in Age with the medians of Sex and Pclass groups
grouped = df.groupby(['Sex', 'Pclass'])['Age'].transform('median')
df['Age'].fillna(grouped, inplace=True)
```

最后，总结下我们的缺失处理策略：
- 为每个特征添加缺失标记，特征选择阶段可以用卡方检验选择性删除
- 有业务含义的进行人工插补
- 缺失率高于20%：删除特征
- 缺失率5-20%：多重插补或条件平均插补
- 缺失率低于5%：简单统计插补

使用pandas实现
```python
# 1. Adds a binary variable to flag missing observations.
def add_missing_indicator(X, y=None, alpha=0.05):
    """
    Adds a binary variable to flag missing observations(one indicator per variable). 
    The added variables (missing indicators) are named with the original variable name plus '_missing'.
    
    Parameters:
    ----------
    alpha: float, default=0.05
        Features with p-values more than alpha are selected.
    """
    # Compute chi-squared stats between each missing indicator and y.
	chi2_stats, p_values = chi2(X.isna(), y)
	# find variables for which indicator should be added.
	missing_indicator = X.iloc[:, p_values > alpha]
	indicator_names = missing_indicator.columns.map(lambda x: x + "_missing")
	X[indicator_names] = missing_indicator
	print(f"Added {len(missing_indicator)} missing indicators")
	return X

# 2. manual imputer
def impute_manually(X):
    """
    Replaces missing values by an arbitrary value
    """
    # boolean
    X.filter(regex="^FLAG_").fillna(0, inplace=True)
    # fill none
    features_fill_none = ["name_type_suite", "occupation_type", "orgnization_type"]
    X[features_fill_none].fillna('None', inplace=True)
    # fill 0
    features_fill_zero = ["obs_30_cnt_social_circle", "AMT_REQ_CREDIT_BUREAU_HOUR"]
    X[features_fill_zero].fillna(0, inplace=True)
    return X

# 3. Remove variables with high missing rate
def drop_missing_data(X, threshold=0.8):
    # Remove variables with missing more than threshold(default 20%)
    threshold = int(X.size * threshold)
    X_new = X.dropna(axis=1, thresh=threshold)
    print(f"Removed {len(X)-len(X_new)} variables with missing more than {threshold:%}")
    return X_new

# 4. conditional statistic completer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import f_classif, chi2
from sklearn.feature_selection import r_regression, f_regression

def fillna_by_groups(X, threshold=0.8, groupby=None, k=2, min_categories=2, bins=10):
    """
    Replaces missing values by groups.

    k: int, default=2
        Number of top features to group by.
    min_categories: int, default=2
        Specifies an lower limit to the number of categories for each feature to group by.
        If None, there is no limit.
    bins: int, default=10
    """
    Y = X.copy()
    variables = Y.isna().mean().lt(1 - threshold).index.to_list()
    
    features_num = X.select_dtypes('number').columns.to_list()
    features_cat = [colname for colname in X.columns if colname not in features_num]
    X[features_num] = X[features_num].apply(pd.qcut, q=bins, duplicates="drop")
    X[features_cat] = X[features_cat].astype('category')
    X = X.transform(lambda x: x.cat.codes)
    X = X.transform(lambda x: x - x.min()) # for chi-squared to stats each non-negative feature
    if groupby is None:
        features_groupby = X.columns.tolist()
    useful_columns = X.nunique().geq(min_categories).index.tolist()
    features_groupby = list(set(features_groupby) & set(useful_columns))
    
    # Estimate mutual information for a target variable.
    for colname in variables:
        other_features = list(set(features_groupby) - {colname})
        if colname in features_num:
            score_func = mutual_info_regression
        elif colname in features_cat:
            score_func = mutual_info_classif
        scores = score_func(X[other_features], Y[colname], discrete_features=True)
        scores = pd.Series(scores, index=other_features).sort_values(ascending=False)
        vars_top_k = scores[:K].index.tolist()
        if colname in features_num:
            # Replaces missing values by the mean or median
            Y[colname] = Y.groupby(vars_top_k)[colname].transform(lambda x:x.fillna(x.median()))
            print(f"Filling the missing values in {colname} with the medians of {vars_top_k} groups.")
        elif colname in features_cat:
            # Replaces missing values by the most frequent category
            Y[colname] = Y.groupby(vars_top_k)[colname].transform(lambda x:x.fillna(x.mode()[0]))
            print(f"Filling the missing values in {colname} with the modes of {vars_top_k} groups.")
    print(f"Transformed {len(variables)} variables with missing (threshold={threshold:.1%}).")
    return Y

# 5. Simple imputer
def impute_simply(X, threshold=0.8):
    """
    Univariate imputer for completing missing values with simple strategies.
    """
    variables = X.isna().mean().lt(1 - threshold).index.to_list()
    features_num = X[variables].select_dtypes('number').columns.to_list()
    features_cat = [colname for colname in variables if colname not in features_num]
    # Replaces missing values by the median or mode
    medians = X[features_num].median().to_dict()
    modes = X[features_cat].apply(lambda x: x.mode()[0]).to_dict()
    impute_dict = {**medians, **modes}
    X[variables] = X[variables].fillna(impute_dict)
    print(f"Transformed {len(variables)} variables with missing (threshold={threshold:.1%}).")
    return X
```

使用sklearn实现

先自定义几个转换器

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X, check_is_fitted

class DropMissingData(BaseEstimator, TransformerMixin):
    """
    Remove features from data.
    
    Parameters
    ----------
    threshold: float, default=None
        Require that percentage of non-NA values in a column to keep it.
    """
    def __init__(self, threshold=0.8):
        if 0 < threshold <= 1:
            self.threshold = threshold
        else:
            raise ValueError("threshold must be a value between 0 < x <= 1. ")
    
    def fit(self, X, y=None):
        """
        Find the rows for which missing data should be evaluated to decide if a
        variable should be dropped.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training data set.

        y: pandas Series, default=None
            y is not needed. You can pass None or y.
        """
        
        # check input dataframe
        X = check_X(X)
        
        # Get the names and number of features in the train set (the dataframe used during fit).
        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]
        
        # Find the features to keep
        self.variables = X.isna().mean().gt(1 - self.threshold).columns.to_list()
        return self
    
    def transform(self, X, y=None):	 
        """
        Remove variables with missing more than threshold.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Returns
        -------
        X_new: pandas dataframe
            The complete case dataframe for the selected variables.
        """
        # Remove variables with missing more than threshold.
        print(f"Removed {len(self.variables)} variables with missing more than {self.threshold:%}")
        return X.drop(self.variables)
    
    def get_feature_names_out(input_features=None):
        """
        Get output feature names for transformation. In other words, returns the
        variable names of transformed dataframe.

        Parameters
        ----------
        input_features : array or list, default=None
            This parameter exits only for compatibility with the Scikit-learn pipeline.

            - If `None`, then `feature_names_in_` is used as feature names in.
            - If an array or list, then `input_features` must match `feature_names_in_`.

        Returns
        -------
        feature_names_out: list
            Transformed feature names.
        """
        check_is_fitted(self)
        
        if input_features is None:
            feature_names_in = self.feature_names_in_
        elif len(input_features) == len(self.n_features_in_):
            # If the input was an array, we let the user enter the variable names.
            feature_names_in = list(input_features)
        else:
            raise ValueError(
                "The number of input_features does not match the number of "
                "features seen in the dataframe used in fit."
                )      
        
        # Remove features.
        feature_names_out = [var for var in self.feature_names_in_ if var not in self.variables]
        return feature_names_out
```

```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import f_classif, chi2
from sklearn.feature_selection import r_regression, f_regression

class ConditionalStatisticImputer(BaseEstimator, TransformerMixin):
    """
    Replaces missing values by groups.
    
    Parameters
    ----------
    threshold: float, default=None
        Require that percentage of non-NA values in a column to impute.
    k: int, default=2
        Number of top features to group by.
    min_categories: int, default=2
        Specifies an lower limit to the number of categories for each feature to group by.
        If None, there is no limit.
    bins: int, default=10
    """
    
    def __init__(self, threshold=0.8, groupby=None, k=2, min_categories=2, bins=10):
        if 0 < threshold <= 1:
            self.threshold = threshold
        else:
            raise ValueError("threshold must be a value between 0 < x <= 1. ")
        self.groupby = groupby 
        self.k = k
        self.min_categories = min_categories
        self.bins = bins
    
    def fit(self, X, y=None):
        """
        Compute correlation coefficient matrix.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The training data set.

        y: pandas Series, default=None
            y is not needed. You can pass None or y.
        """
        
        # check input dataframe
        X = check_X(X)
        
        # Get the names and number of features in the train set (the dataframe used during fit).
        self.feature_names_in_ = X.columns.to_list()
        self.n_features_in_ = X.shape[1]
        
        Y = X.copy()
        self.variables = Y.isna().mean().lt(1 - threshold).index.to_list()
        X = self.discretize_encode(X, y=None, bins=self.bins)
        
        if self.groupby is None:
            features_groupby = X.columns.to_list()
        else:
            features_groupby = list(groupby)
        useful_columns = X.nunique().geq(self.min_categories).index.to_list()
        features_groupby = list(set(features_groupby) & set(useful_columns))
        
        score_matrix = pd.DataFrame(index=self.variables) # init a matrix to hold scores
        # Estimate mutual information for a target variable.
        for colname in self.variables:
            if colname in features_num:
                score_func = mutual_info_regression
            elif colname in features_cat:
                score_func = mutual_info_classif
            scores = score_func(X[features_groupby], Y[colname], discrete_features=True)
            score_matrix[colname] = pd.Series(scores, index=features_groupby).sort_values(ascending=False)
        self.score_matrix  = score_matrix 
        return self

    def transform(self, X, y=None):	 
        """
        Remove variables with missing more than threshold.

        Parameters
        ----------
        X: pandas dataframe of shape = [n_samples, n_features]
            The dataframe to be transformed.

        Returns
        -------
        X_new: pandas dataframe
            The complete case dataframe for the selected variables.
        """
        Y = X.copy()
        X = self.discretize_encode(X, y=None, bins=self.bins)
        features_num = X.select_dtypes('number').columns.to_list()
        features_cat = [colname for colname in X.columns if colname not in features_num]
        
        for colname in self.variables:  
            vars_top_k = self.score_matrix[colname].drop(colname)[:self.K].index.to_list()
            if colname in features_num:
                # Replaces missing values by the mean or median
                Y[colname] = Y.groupby(vars_top_k)[colname].transform(lambda x:x.fillna(x.median()))
                print(f"Filling the missing values in {colname} with the medians of {vars_top_k} groups.")
            elif colname in features_cat:
                # Replaces missing values by the most frequent category
                Y[colname] = Y.groupby(vars_top_k)[colname].transform(lambda x:x.fillna(x.mode()))
                print(f"Filling the missing values in {colname} with the modes of {vars_top_k} groups.")
        print(f"Transformed {len(variables)} variables with missing (threshold={threshold:.1%}).")
        return Y
    
    def discretize_encode(self, X, y=None, bins=10):
        features_num = X.select_dtypes('number').columns.to_list()
        features_cat = [colname for colname in X.columns if colname not in features_num]
        
        X[features_num] = X[features_num].apply(pd.qcut, q=bins, duplicates="drop")
        X[features_cat] = X[features_cat].astype('category')
        X = X.transform(lambda x: x.cat.codes)
        X = X.transform(lambda x: x - x.min()) # for chi-squared to stats each non-negative feature
        return X
        
    def get_feature_names_out(input_features=None):
        """
        Get output feature names for transformation. In other words, returns the
        variable names of transformed dataframe.

        Parameters
        ----------
        input_features : array or list, default=None
            This parameter exits only for compatibility with the Scikit-learn pipeline.

            - If `None`, then `feature_names_in_` is used as feature names in.
            - If an array or list, then `input_features` must match `feature_names_in_`.

        Returns
        -------
        feature_names_out: list
            Transformed feature names.
        """
        check_is_fitted(self)
        
        if input_features is None:
            feature_names_in = self.feature_names_in_
        elif len(input_features) == len(self.n_features_in_):
            # If the input was an array, we let the user enter the variable names.
            feature_names_in = list(input_features)
        else:
            raise ValueError(
                "The number of input_features does not match the number of "
                "features seen in the dataframe used in fit."
                )      
        
        return feature_names_in
```


整合到pipeline中
```python
from sklearn.impute import MissingIndicator
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.pipeline import ColumnUnion, make_union, make_pipeline, make_union
from feature_selection import SelectKBest, SelectFpr

def select_column(X)
    fill_dict = dict()
    fill_dict['zero'] = X.columns.str.startswith("FLAG_").to_list() + ["obs_30_cnt_social_circle", "AMT_REQ_CREDIT_BUREAU_HOUR"]
    fill_dict['None'] = ["name_type_suite", "occupation_type", "orgnization_type"]
    fill_dict['median'] = X.select_dtypes("number").isna().mean().lt(0.05).index.to_list()
    fill_dict['mode'] = X.select_dtypes("category").isna().mean().lt(0.05).index.to_list()
    return fill_dict

fill_dict = select_column(X_train)

imputer = make_column_transformer(
    (SimpleImputer(strategy="constant", fill_value="None"), features_fill_none)),
    (SimpleImputer(strategy="constant", fill_value=0), features_fill_zero),
    (SimpleImputer(strategy="median"), features_fill_median),
    (SimpleImputer(strategy="most_frequent"), features_fill_mode),
    remainder=KNNImputer(),
    verbose=True)

# find variables for which indicator should be added.
indicate_missing = make_pipeline(
    MissingIndicator(features='all', sparse=False),
    SelectFpr(alpha=0.05)
    verbose=True)

handle_missing_pipeline = make_union(imputer, indicate_missing, verbose=True)
```

最后确认缺失值是否已全部处理完毕：
```python
X.isna().sum().max()
```

## 标准化/归一化

数据标准化和归一化可以提高一些算法的准确度，也能加速梯度下降收敛速度。也有不少模型不需要做标准化和归一化，主要是基于概率分布的模型，比如决策树大家族的CART，随机森林等。

- **z-score标准化**是最常见的特征预处理方式，基本所有的线性模型在拟合的时候都会做标准化。前提是假设特征服从正态分布，标准化后，其转换成均值为0标准差为1的标准正态分布。
- **max-min标准化**也称为离差标准化，预处理后使特征值映射到[0,1]之间。这种方法的问题就是如果测试集或者预测数据里的特征有小于min，或者大于max的数据，会导致max和min发生变化，需要重新计算。所以实际算法中， 除非你对特征的取值区间有需求，否则max-min标准化没有 z-score标准化好用。
- **L1/L2范数标准化**：如果我们只是为了统一量纲，那么通过L2范数整体标准化。

| sklearn.preprocessing | 说明                                  |
| --------------------- | ------------------------------------- |
| StandardScaler()      | z-score标准化                         |
| Normalizer(norm='l2') | 使用`l1`、`l2`或`max`范数归一化       |
| MinMaxScaler()        | min-max归一化                         |
| MaxAbsScaler()        | Max-abs归一化，缩放稀疏数据的推荐方法 |
| RobustScaler()        | 分位数归一化，推荐缩放有离群值的数据      |

pandas实现z-score标准化和分位数归一化在之前检测离群值函数里已有，其他标准化方法不太常用。

由于数据集中依然存在一定的离群点，我们可以用RobustScaler对数据进行标准化处理。
```python
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
```

## 正态变换

我们先来计算一下各个数值特征的偏度：

```python
# Check the skew of all numerical features
skewness = X_train.select_dtypes('number').skew()
skewness = skewness[abs(skewness) > 0.75].sort_values()
```

可以看到这些特征的偏度较高，在许多回归算法中，尤其是线性模型，常常假设数值型特征服从正态分布。因此我们尝试变换，让数据接近正态分布。

以age特征为例，我们画出分布图和QQ图（使用之前定义的函数）。
> Quantile-Quantile图是一种常用的统计图形，用来比较两个数据集之间的分布。它是由标准正态分布的分位数为横坐标，样本值为纵坐标的散点图。如果QQ图上的点在一条直线附近，则说明数据近似于正态分布，且该直线的斜率为标准差，截距为均值。

```python
norm_comparison_plot(np.log1p(df['SalePrice']))
plt.show()
```

sklearn.preprocessing模块目前支持的非线性变换：

| 方法         | 说明    |
| ------------ | -------------- | 
|QuantileTransformer|分位数变换，映射到[0,1]之间的均匀分布，或正态分布|
|PowerTransformer| 幂变换，将数据从任何分布映射到尽可能接近高斯分布，以稳定方差并最小化倾斜度|

此外，最常用的是log变换。对于含有负数的特征，可以先min-max缩放到[0,1]之间后再做变换。

这里我们对age特征做Box-Cox变换

```python
# Box Cox Transformation of skewed features (instead of log-transformation)
norm_trans = PowerTransformer("box-cox")
```

```python
norm_comparison_plot(np.log1p(df['SalePrice']))
plt.show()
```

可以看到经过Box-Cox变换后，基本符合正态分布了。

## Baseline

至此，数据预处理已经基本完毕，我们可以选择模型开始训练了。为了检测模型的表现，我们先定义一个评估函数。

```python
def score_dataset(X, y, model):
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    log_y = np.log(y)
    score = cross_val_score(
        model, X, log_y, cv=5, scoring="neg_root_mean_squared_error",
    ).mean()
    return -1 * score

X = df.drop(ID_col).copy()
y = X.pop(target)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, randm_state=SEED)
baseline_score = score_dataset(X, y)
print(f"Baseline score: {baseline_score:.5f} RMSLE")

categorical_cols = df.select_dtypes(["object", "category"]).columns.tolist()
numeric_cols = df.select_dtypes("number").columns.tolist()
```

使用sklearn实现预处理的代码如下
```python
# split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)

cleaner = FunctionTransformer(clean, features_name_out='one-to-one')

prepare_data = Pipeline([
    ("clean", cleaner),
    ("encode", categorical_encoder),
    mean_encoder
    handle_missing_pipeline
    ("scale", RobustScaler()),
    ("classifier", LogitRegression())
])
categorical_encoder 
estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA())]
combined = FeatureUnion(estimators)
```

选择逻辑回归、SVM和LightGBM模型训练结果作为baseline

```python
```
|auc  |  |SVM|
|--|--|--|
| baseline |  ||



