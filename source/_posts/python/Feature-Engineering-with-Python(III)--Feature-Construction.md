---
title: 特征工程(I)--特征构造
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

# 特征构造

特征构造是从现有数据创建新特征的过程。目标是构建有用的功能，帮助我们的模型了解数据集中的信息与给定目标之间的关系。

## 简单数学变换

我们可以根据业务含义，创建具有一些明显实际含义的补充特征，例如：

```python
df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'] # 贷款金额相对于收入的比率
df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']  # 贷款年金占总收入比率
df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'] # 以月为单位的付款期限
df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'] #工作时间占年龄的比率
df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'] # 该用户家庭的人均收入
```

我们可以在图形中直观地探索这些新变量：

```python
plt.figure(figsize = (12, 20))
# iterate through the new features
for i, feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):
    # create a new subplot for each source
    plt.subplot(4, 1, i + 1)
    # plot repaid loans
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0, feature], label = 'target == 0')
    # plot loans that were not repaid
    sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1, feature], label = 'target == 1')
    
    # Label the plots
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)
```

当然，我们不可能手动计算出所有有实际含义的数学特征。我们可以借助于 featuretools 包找到尽可能多的特征组合进行加减乘除运算。

```python
# Make an entityset and add the entity
es = ft.EntitySet(id = 'app')
es.entity_from_dataframe(entity_id = 'data', dataframe = df, index = ID_col)

# Run deep feature synthesis with transformation primitives
feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'data',
    trans_primitives = ['add_numeric', 'multiply_numeric'])
feature_matrix.head()
```

另外，还有对单变量的函数运算：包括对数、指数、倒数、平方根、三角函数等。

```python
log_transformer = FunctionTransformer(func=np.log1p, validate=True, accept_sparse=True,
    feature_names_out=lambda x: f"log1p({x})")

exp_transformer = FunctionTransformer(func=np.exp, validate=True, accept_sparse=True,
    feature_names_out=lambda x: f"exp({x})")

reciprocal_transformer = FunctionTransformer(func=np.exp, validate=True, accept_sparse=True,
    feature_names_out=lambda x: f"1/{x}")

sqrt_transformer = FunctionTransformer(func=np.exp, validate=True, accept_sparse=True,
    feature_names_out=lambda x: f"sqrt({x})")

sin_transformer = FunctionTransformer(func=np.sin, validate=True, accept_sparse=True,
    feature_names_out=lambda x: f"sin({x})")

math_transformer = make_union(
    make_pipeline(MinMaxscaler(),  log_transformer),
    exp_transformer,
    reciprocal_transformer,
    make_pipeline(MinMaxscaler(),  sqrt_transformer),
    sin_transformer
)
```




## 分组统计特征构造

分组统计特征衍生，顾名思义，就是分类特征和连续特征间的分组交互统计，这样可以得到更多有意义的特征，例如：

```python
# Group loans by client id and calculate mean, max, min of loans
stats = loans.groupby('client_id')['loan_amount'].agg(['mean', 'max', 'min'])
stats.columns = ['mean_loan_amount', 'max_loan_amount', 'min_loan_amount']

# Merge with the clients dataframe
stats = clients.merge(stats, left_on = 'client_id', right_index=True, how = 'left')

stats.head(10)
```

常用的统计量：

- mean/var: 均值、方差
- median: 中位数
- max/min: 最大值、最小值
- skew: 偏度
- mode: 众数
- nunique: 类别数
- count: 个数
- quantile：分位数

> 注意：分组特征必须是离散特征，且最好是一些取值较多的离散变量，这样可以避免新特征出现大量重复取值。分组使用连续值特征时一般需要先进行离散化。

接下来我们定义2个转换器：`AggregateNumericCreator` 和 `AggregateCategoricalCreator`分别用来处理数值类型和分类型的分组变量衍生。

```python
AggregateNumericCreator
median()_by()

# Function for Numeric Aggregations
def agg_numeric(df, group_var, df_name):
    """
    Aggregates the numeric values in a dataframe. This can
    be used to create features for each instance of the grouping variable.
    
    Parameters
    --------
        df (dataframe): 
            the dataframe to calculate the statistics on
        group_var (string): 
            the variable by which to group df
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated for 
            all numeric columns. Each instance of the grouping variable will have 
            the statistics (mean, min, max, sum; currently supported) calculated. 
            The columns are also renamed to keep track of features created.
    
    """
    # Remove id variables other than grouping variable
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()

    # Need to create new column names
    columns = [group_var]

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))

    agg.columns = columns
    
    # Remove the columns with all redundant values
    _, idx = np.unique(agg, axis = 1, return_index=True)
    agg = agg.iloc[:, idx]
    
    return agg
  
bureau_agg_new = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_agg_new.head()
```



```python
# Function to Handle Categorical Variables
def agg_categorical(df, group_var, df_name):
    """
    Computes counts and normalized counts for each observation
    of `group_var` of each unique category in every categorical variable.
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    group_var : string
        The variable by which to group the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns
    
    Return
    --------
    categorical : dataframe
        A dataframe with counts and normalized counts of each unique category in every categorical variable
        with one row for every unique value of the `group_var`.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))

    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    categorical.columns = column_names

    # Remove duplicate columns by values
    _, idx = np.unique(categorical, axis = 1, return_index = True)
    categorical = categorical.iloc[:, idx]
    
    return categorical

bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_counts.head()
```



## 特征交互

通过将单独的特征求笛卡尔乘积的方式来组合2个或更多个特征，从而构造出组合特征。最终获得的预测能力将远远超过任一特征单独的预测能力。

```python
X = pd.get_dummies(df.BldgType, prefix="Bldg")  # 住宅类型
X = X.mul(df.GrLivArea, axis=0) # 居住面积
```

笛卡尔乘积组合特征方法一般应用于类别特征之间，连续值特征使用笛卡尔乘积组合特征时一般需要先进行离散化。

```python
FeatureInteraction()

def cartesian_product_feature_crosses(df, feature1_name, feature2_name):
    feature1_df = pd.get_dummies(df[feature1_name], prefix=feature1_name)
    feature1_columns = feature1_df.columns

    feature2_df = pd.get_dummies(df[feature2_name], prefix=feature2_name)
    feature2_columns = feature2_df.columns

    combine_df = pd.concat([feature1_df, feature2_df], axis=1)

    crosses_feature_columns = []
    for feature1 in feature1_columns:
        for feature2 in feature2_columns:
            crosses_feature = '{}&{}'.format(feature1, feature2)
            crosses_feature_columns.append(crosses_feature)

            combine_df[crosses_feature] = combine_df[feature1] * combine_df[feature2]

    combine_df = combine_df.loc[:, crosses_feature_columns]
    return combine_df

combine_df = cartesian_product_feature_crosses(df, 'color', 'light')
display(combine_df.head())
```

利用笛卡尔乘积的方法来构造组合特征这种方法虽然简单，但麻烦的是会使得特征数量爆炸式增长。一个可以取N个不同值的类别特征，与一个可以去M个不同值的类别特征做笛卡尔乘积，就能构造出N*M个组合特征。



## 多项式特征

多项式特征是 sklearn 中特征构造的最简单方法。当我们创建多项式特征时，我们希望避免使用过高的度数，这是因为特征的数量随着度数指数级地变化，并且可能过拟合。

我们可以使用3度多项式来查看结果：

```python
from sklearn.preprocessing import PolynomialFeatures

# Make a new dataframe for polynomial features
poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
 
# imputer for handling missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')
 
poly_target = poly_features['TARGET']
poly_features = poly_features.drop(columns = ['TARGET'])
 
# Need to impute missing values
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)
 
from sklearn.preprocessing import PolynomialFeatures
                                  
# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 3)

# Train the polynomial features
poly_transformer.fit(poly_features)
 
# Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)

poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]

# Print out the new shapes
print('Training data with polynomial features shape: ', app_train_poly.shape)
```

## 聚类分析

聚类算法在特征构造中的应用有不少，例如：利用聚类算法对文本聚类，使用聚类类标结果作为输入特征；利用聚类算法对单个数值特征进行聚类，相当于使用聚类算法进行特征分箱；利用聚类算法对R、F、M数据进行聚类，类似RFM模型，然后再使用代表衡量客户价值的聚类类标结果作为输入特征。

当一个或多个特征具有多峰分布（有两个或两个以上清晰的峰值）时，可以使用聚类算法为每个峰值分类，并输出聚类类标结果。一般聚类类标结果为一个数值，但实际上这个数值并没有大小之分，所以一般需要进行特征编码。可以使用one-hot编码，或者创建新特征用来度量样本和每个类中心的相似性（距离）。

相似性度量通常使用径向基函数(RBF)来计算（任何只依赖于输入值与不动点之间距离的函数）。最常用的RBF是高斯RBF，其输出值随着输入值远离固定点而呈指数衰减。高斯RBF可以使用Scikit-Learn的rbf_kernel()函数计算 $k(x,y)=\exp(-\gamma\|x-y\|^2)$，超参数 gamma 确定当x远离y时相似性度量衰减的速度。

例如，您可以创建一个新的高斯RBF特征来测量房屋中位年龄和35：

```python
from sklearn.metrics.pairwise import rbf_kernel
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
```

下图显示了这一新特征作为住房中位数年龄的函数(实线)。如图所示，新的年龄相似性特征在35岁时达到峰值，如果这个特定的特征与较低的价格有很好的相关性，那么这个新特征将有很好的机会发挥作用。

```python
```

如果你给rbf_kernel()函数传递一个有两个特征的数组，它会测量二维距离(欧几里得)来测量相似性。

接下来，我们自定义一个转换器，该转换器在fit()方法中使用KMeans聚类器来识别训练数据中的主要聚类，然后在transform()方法中使用`rbf_kernel()` 来衡量每个样本与每个聚类中心的相似程度：
```python
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
    def fit(self, X, y=None, sample_weight=None):
        X_scaled = standscaler()
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self 
    def transform(self, X):
    	return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    def get_feature_names_out(self, input_features=None):
    	return [f"Centroid_{i}" for i in range(self.n_clusters)]
```

现在让我们使用这个转换器：
```python
cluster_features = [
    "LotArea",
    "TotalBsmtSF",
    "FirstFlrSF",
    "SecondFlrSF",
    "GrLivArea",
]

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]],
sample_weight=housing_labels)
```

## 主成分分析

由于我们新增的这些特征都是和原始特征高度相关，可以使用PCA的主成分作为新的特征，消除相关性。

查看相关系数矩阵

```python
def apply_pca(X, standardize=True):
    # Standardize
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs

def pca_components(df, features):
    X = df.loc[:, features]
    _, X_pca, _ = apply_pca(X)
    return X_pca


pca_features = [
    "GarageArea",
    "YearRemodAdd",
    "TotalBsmtSF",
    "GrLivArea",
]
```

## featuretools

在许多情况下数据分布在多个表中。而机器学习模型必须用单个表进行训练，因此特征工程要求我们将所有数据汇总到一个表中。

本项目数据由Home Credit（捷信）提供，Home Credit致力于向无银行账户的人群提供信贷，是中东欧以及亚洲领先的消费金融提供商之一。在中东欧（CEE），俄罗斯，独联体国家（CIS）和亚洲各地服务于消费者。

Home Credit有三类产品，信用卡，POS（消费贷），现金贷，三类产品的英文分别是：Revolving loan (credit card)，Consumer installment loan (Point of sales loan – POS loan)，Installment cash loan。三类产品逻辑是这样的：

1. 首先提供POS，入门级产品，类似于消费贷。只能用于消费，金额有限，风险最小，客户提供的信息也最少。
2. 然后是credit card，在本次竞赛中称为revolving loan。循环授信，主要用于消费。
3. 最后才是cash loan，用户能得到现金，风险最大。

数据集包括了8个不同的数据文件，大致可以分为三大类：

- application_{train|test}：包含每个客户社会经济信息和Home Credit贷款申请信息的主要文件。每行代表一个贷款申请，由SK_ID_CURR唯一标识。训练集30.75万数据，测试集4.87万数据。其中训练集中`TARGET=1`表示未偿还贷款。通过这两个文件，就能对这个任务做基本的数据分析和建模，也是本篇的主要内容。

用户在征信机构的历史征信记录可以用来作为风险评估的参考，但是征信数据往往不全，因为这些人本身就很少有银行记录。数据集中bureau.csv和 bureau_balance.csv 对应这部分数据。

- bureau：征信机构提供的客户之前在其他金融机构的贷款申请数据。一个用户（SK_ID_CURR）可以有多笔贷款申请数据（SK_ID_BUREAU）。总计171万数据。
- bureau_balance：征信机构统计的之前每笔贷款（SK_ID_BUREAU）的每月（MONTHS_BALANCE）的还款欠款记录。共有2729万条数据。

数据集中previous_application.csv, POS_CASH_balance.csv，credit_card_balance.csv，installments_payment.csv这部分数据是来自Home Credit产品的历史使用信息。信用卡在欧洲和美国很流行，但在以上这些国家并非如此。所以数据集中信用卡数据偏少。POS只能买东西，现金贷可以得到现金。三类产品都有申请和还款记录。

- previous_application：该表是客户在申请这次贷款之前的申请记录。一个用户（SK_ID_CURR）可以有多笔历史数据（SK_ID_PREV）。共计167万条。
- POS_CASH_balance：以前每月pos流水记录。由SK_ID_PREV和 months_balance唯一标识，共计1000万条数据。
- credit_card_balance：每月信用卡账单表。由MONTHS_BALANCE和SK_ID_PREV唯一标识，合计384万条数据。
- installments_payment：分期付款表。由 SK_ID_PREV, name_instalment_version, name_instalment_number 唯一标识，共计1360万条数据。

数据集的字段描述性文件：[HomeCredit_columns_description.csv](/ipyna/HomeCredit_columns_description.csv)

我们对这些数据集，通常基于 id 值连接多个表，然后计算数值特征的一些统计量。featuretools 包能很方便的自动完成这些任务。

featuretools 涉及到3个概念：实体(entity)、关系(relationship)和算子(primitive)。

- 所谓的实体就是一张表或者一个dataframe，多张表的集合就叫实体集(entityset)。
- 关系就是表之间的关联键的定义。
- 而算子就是一些特征工程的函数。有应用于实体集的聚合操作(Aggregation primitives)和应用于单个实体的转换操作(Transform primitives)两种。

```python
import pandas as pd

primitives = ft.list_primitives()

# Load datasets
app_train = pd.read_csv('data/application_train.csv')
app_test = pd.read_csv('data/application_test.csv')
bureau = pd.read_csv('data/bureau.csv')
bureau_balance = pd.read_csv('data/bureau_balance.csv')
cash = pd.read_csv('data/POS_CASH_balance.csv')
credit = pd.read_csv('data/credit_card_balance.csv')
previous = pd.read_csv('data/previous_application.csv')
installments = pd.read_csv('data/installments_payments.csv')
app_test['TARGET'] = np.nan

# Join together training and testing
app = app_train.append(app_test, ignore_index = True, sort = True)
 
# fill all NaN values with zero so they do not hinder with the processing
app.fillna(0, inplace=True)
bureau.fillna(0, inplace=True)
bureau_balance.fillna(0, inplace=True)
cash.fillna(0, inplace=True)
credit.fillna(0, inplace=True)
previous.fillna(0, inplace=True)
installments.fillna(0, inplace=True)
# Empty entity set with id applications
es = ft.EntitySet(id = 'clients')

# Entities with a unique index
es = es.add_dataframe(dataframe_name= 'app', dataframe = app, 
index = 'SK_ID_CURR')
es = es.add_dataframe(dataframe_name= 'bureau', dataframe = bureau, 
index = 'SK_ID_BUREAU')
es = es.add_dataframe(dataframe_name= 'previous', dataframe = previous, 
index = 'SK_ID_PREV')
# Entities that do not have a unique index
es = es.add_dataframe(dataframe_name= 'bureau_balance', dataframe = bureau_balance, 
    make_index = True, index = 'bureaubalance_index')
es = es.add_dataframe(dataframe_name= 'cash', dataframe = cash, 
    make_index = True, index = 'cash_index')
es = es.add_dataframe(dataframe_name= 'installments', dataframe = installments,
    make_index = True, index = 'installments_index')
es = es.add_dataframe(dataframe_name= 'credit', dataframe = credit,
    make_index = True, index = 'credit_index')
    # Relationship between app_train and bureau
es = es.add_relationship('app', 'SK_ID_CURR', 'bureau', 'SK_ID_CURR')
es = es.add_relationship('bureau', 'SK_ID_BUREAU', 'bureau_balance', 'SK_ID_BUREAU')
es = es.add_relationship('app','SK_ID_CURR', 'previous', 'SK_ID_CURR')
es = es.add_relationship('previous', 'SK_ID_PREV', 'cash', 'SK_ID_PREV')
es = es.add_relationship('previous', 'SK_ID_PREV', 'installments', 'SK_ID_PREV')
es = es.add_relationship('previous', 'SK_ID_PREV', 'credit', 'SK_ID_PREV')

print(es)
# Default primitives from featuretools
agg_primitives =  ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
trans_primitives =  ["day", "year", "month", "weekday", "haversine", 
"num_words", "num_characters"]

# DFS with specified primitives
feature_matrix, feature_defs = ft.dfs(entityset = es, 
target_dataframe_name = 'app',
    trans_primitives = trans_primitives,
    agg_primitives=agg_primitives,
    max_depth = 4, n_jobs = -1, verbose = 1)

# view first 10 features
print(feature_defs[:10])
```


## 小结

```python
feature_creator = make_pipeline(
)


def create_features(df, df_test=None):
    X = df.copy()
    y = X.pop("SalePrice")
    mi_scores = make_mi_scores(X, y)

    # Combine splits if test data is given
    #
    # If we're creating features for test set predictions, we should
    # use all the data we have available. After creating our features,
    # we'll recreate the splits.
    if df_test is not None:
        X_test = df_test.copy()
        X_test.pop("SalePrice")
        X = pd.concat([X, X_test])

    # Lesson 2 - Mutual Information
    X = drop_uninformative(X, mi_scores)

    # Lesson 3 - Transformations
    X = X.join(mathematical_transforms(X))
    X = X.join(interactions(X))
    X = X.join(counts(X))
    # X = X.join(break_down(X))
    X = X.join(group_transforms(X))

    # Lesson 4 - Clustering
    # X = X.join(cluster_labels(X, cluster_features, n_clusters=20))
    # X = X.join(cluster_distance(X, cluster_features, n_clusters=20))

    # Lesson 5 - PCA
    X = X.join(pca_inspired(X))
    # X = X.join(pca_components(X, pca_features))
    # X = X.join(indicate_outliers(X))

    X = label_encode(X)

    # Reform splits
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)

    # Lesson 6 - Target Encoder
    encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
    X = X.join(encoder.fit_transform(X, y, cols=["MSSubClass"]))
    if df_test is not None:
        X_test = X_test.join(encoder.transform(X_test))

    if df_test is not None:
        return X, X_test
    else:
        return X


df_train, df_test = load_data()
X_train = create_features(df_train)
y_train = df_train.loc[:, "SalePrice"]

score_dataset(X_train, y_train)
```

