---
title: 特征工程(I)--特征选择
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

# 特征选择

## 摘要

我们现在已经有大量的特征可使用，有的特征携带的信息丰富，有的特征携带的信息有重叠，有的特征则属于无关特征，尽管在拟合一个模型之前很难说哪些特征是重要的，但如果所有特征不经筛选地全部作为训练特征，经常会出现维度灾难问题，甚至会降低模型的泛化性能（因为较无益的特征会淹没那些更重要的特征）。因此，我们需要进行特征筛选，排除无效/冗余的特征，把有用的特征挑选出来作为模型的训练数据。

特征选择方法有很多，一般分为三类：

- 过滤法（Filter）比较简单，它按照特征的发散性或者相关性指标对各个特征进行评分，设定评分阈值或者待选择阈值的个数，选择合适特征。
- 包装法（Wrapper）根据目标函数，通常是预测效果评分，每次选择部分特征，或者排除部分特征。
- 嵌入法（Embedded）则稍微复杂一点，它先使用选择的算法进行训练，得到各个特征的权重，根据权重从大到小来选择特征。


| sklearn.feature_selection | 所属方法 | 说明                                                   |
| ------------------------- | -------- | ------------------------------------------------------ |
| VarianceThreshold         | Filter   | 方差选择法                                             |
| SelectKBest               | Filter   | 常用相关系数、卡方检验、互信息、IV值作为得分计算的方法 |
|SelectPercentile|Filter|根据最高分数的百分位数选择特征|
|SelectFpr, SelectFdr, SelectFwe|Filter|根据假设检验的p-value选择特征|
| RFECV                       | Wrapper  | 在交叉验证中执行递归式特征消除                                  |
| SequentialFeatureSelector | Wrapper  | 前向/向后搜索                                          |
| SelectFromModel           | Embedded | 训练基模型，选择权值系数较高的特征                     |

## 单变量特征选择

Relief（Relevant Features）是著名的过滤式特征选择方法。该方法假设特征子集的重要性是由子集中的每个特征所对应的相关统计量分量之和所决定的。所以只需要选择前k个大的相关统计量对应的特征，或者大于某个阈值的相关统计量对应的特征即可。

常用的指标过滤方法：

| 函数                      | python模块                           | 说明                                       |
| :------------------------ | :----------------------------------- | ------------------------------------------ |
| VarianceThreshold         | sklearn.feature_selection            | 方差过滤                                   |
| r_regression              | sklearn.feature_selection            | 回归任务的目标/特征之间的Pearson相关系数。 |
| f_regression              | sklearn.feature_selection            | 回归任务的目标/特征之间的t检验F值。        |
| mutual_info_regression    | sklearn.feature_selection            | 估计连续目标变量的互信息。                 |
| chi2                      | sklearn.feature_selection            | 分类任务的非负特征的卡方值和P值。          |
| f_classif                 | sklearn.feature_selection            | 分类任务的目标/特征之间的方差分析F值。     |
| mutual_info_classif       | sklearn.feature_selection            | 估计离散目标变量的互信息。                 |
| df.corr, df.corrwith      | pandas                               | Pearson, Kendall, Spearman相关系数         |
| calculate_mi_scores             | self-define                          | 互信息                                     |
| calculate_iv_scores              | self-define                          | IV值                                       |
| calculate_gini_scores             | self-define                          | 基尼系数                                   |
| variance_inflation_factor | statsmodels.stats.outliers_influence | VIF值                                      |
| df.isna().mean()          | pandas                               | 缺失率                                     |


Utilities特征有两个缺失值，且只有一个样本是“NoSeWa”，除此之外全部都是“AllPub”，因此该项特征的方差非常小，我们可以直接将其删去。

```python
df = df.drop(['Utilities'], axis=1)
```

减少特征数量的方法有很多，在这里我们将通过三种方法：

1. 删除共线变量
2. 删除缺少许多值的变量
3. 使用特征重要性只保留“重要”变量 

皮尔森相关系数是一种最简单的，能帮助理解特征和响应变量之间关系的方法，该方法衡量的是变量之间的线性相关性。

*了识别高度相关的变量，我们可以计算数据中每个变量与所有其他变量的相关性（这是一个相当昂贵的计算过程）！然后，我们选择相关矩阵的上三角形，并根据阈值从每对高度相关变量中删除一个变量。这在以下代码中实现：*

```python
# Function to calculate correlations with the target for a dataframe

# Threshold for removing correlated variables
threshold = 0.9

# Absolute value correlation matrix
corr_matrix = app.corr().abs()

# Upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

# Remove the columns
app = app.drop(columns = to_drop)


# Empty dictionary to hold correlated variables
above_threshold_vars = {}
```

在这个实现中，我使用0.9的相关系数阈值来删除共线变量。因此，对于相关性大于0.9的每对特征，我们删除其中一对特征。**在1465个总功能中，这删除了583个，**表明我们创建的许多变量是多余的。



相关系数主要用于评估两个连续变量间的相关性。

```python
# Find correlations with the target and sort
df.corrwith(target)
# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

sklearn.feature_selection import SelectKBest, r_regression
SelectKBest(r_regression)

# Extract the EXT_SOURCE variables and show correlations
ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
ext_data_corrs = ext_data.corr()
ext_data_corrs
```
方差分析主要用于分类问题中连续特征的相关性。
f_classif  

卡方检验是一种用于衡量两个分类变量之间相关性的统计方法。
如果针对分类问题，f_classif和chi2两个评分函数搭配使用，就能够完成一次完整的特征筛选，其中f_classif用于筛选连续特征，chi2用于筛选离散特征。

```python
sklearn.feature_selection import chi2
```

互信息是从信息熵的角度分析各个特征和目标之间的关系包括线性和非线性关系）。此处可以使用均值*0.1作为阈值进行筛选

```python
sklearn.feature_selection import SelectKBest, mutual_info_regression

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

X = df_train.copy()
y = X.pop("SalePrice")

mi_scores = make_mi_scores(X, y)
mi_scores
```



IV（Information Value）用来评价分箱特征对二分类变量的预测能力。一般认为IV小于0.02的特征为无用特征。

```python
def calculate_iv_scores(X, y, discrete_features=None, bins=10):
	X = pd.DataFrame(X)
    y = pd.Series(y)
    assert (y.nunique() != 2), "y must be binary"
    iv_scores = pd.Series()
    if discrete_features is None:
        discrete_features = X.select_dtypes(["object", "category"]).columns
    for colname in X.columns:
        if colname in discrete_features:
            var = X[colname]
        else:
            var = pd.qcut(X[colname], bins, duplicates="drop")
        grouped = y.groupby(var).agg([('Positive','sum'),('All','count')]) 
        grouped['Negative'] = grouped['All']-grouped['Positive'] 
        grouped['Positive rate'] = grouped['Positive']/grouped['Positive'].sum()
        grouped['Negative rate'] = grouped['Negative']/grouped['Negative'].sum()
        grouped['woe'] = np.log(grouped['Positive rate']/grouped['Negative rate'])
        grouped['iv'] = (grouped['Positive rate']-grouped['Negative rate'])*grouped['woe']
        grouped.name = colname 
        iv_scores[colname] = grouped['iv'].sum()
    return iv_scores
```

基尼系数用来衡量分类问题中特征对目标变量的影响程度。它的取值范围在0到1之间，值越大表示特征对目标变量的影响越大。常见的基尼系数阈值为0.02，如果基尼系数小于此阈值，则被认为是不重要的特征。


```python
from sklearn.metrics import roc_auc_score

def gini_score(df, feature, target):
x = df[feature]
y = df[target]
gini = 2 * roc_auc_score(y, x) - 1
return gini

def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


gini_predictions = gini(actual, predictions)
ngini= gini_normalized(actual, predictions)
```



  VIF用于衡量特征之间的共线性程度。通常，VIF小于5被认为不存在多重共线性问题，VIF大于10则存在明显的多重共线性问题。


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
 
df = pd.DataFrame(
    {'a': [1, 1, 2, 3, 4],
     'b': [2, 2, 3, 2, 1],
     'c': [4, 6, 7, 8, 9],
     'd': [4, 3, 4, 5, 4]}
)
 
X = add_constant(df)

>>> pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns)


vif_feats = calc_vif(df.drop(['target'], axis=1))

high_vif_feats = vif_feats[vif_feats['VIF'] > 10]['feature'].tolist()

print("Features with high VIF:", high_vif_feats)
```

- 消除两两高度相关的特征 # Remove Collinear Variables
- 消除多特征间的共线性 Collinear Features
- 



删除无效特征

```python
def drop_uninformative(df, mi_scores):
    return df.loc[:, mi_scores > 0.0]

X = df_train.copy()
y = X.pop("SalePrice") 
X = drop_uninformative(X, mi_scores)

score_dataset(X, y)
```

## 递归特征消除

最常用的包装法是递归消除特征法(recursive feature elimination,以下简称RFE)。递归消除特征法使用一个机器学习模型来进行多轮训练，每轮训练后，消除最不重要的特征，再基于新的特征集进行下一轮训练。

总特征个数
rfe.n_features_in_
筛选后特征个数
rfe.n_features_

use linear regression as the model

rank all features, i.e continue the elimination until the last one

Features sorted by their rank:


## 特征重要性

嵌入法也是用模型来选择特征，但是它和RFE的区别是它不是通过不停的筛掉特征来进行训练，而是使用的都是特征全集。

- 最常用的是使用带惩罚项（$\ell_1,\ell_2$ 正则项）的基模型，来选择特征，例如 Lasso，Ridge。
- 或者简单的训练基模型，选择权重较高的特征。

训练随机森林模型，并通过feature_importances_属性获取每个特征的重要性分数。

删除重要性为0的功能是一个非常安全的选择，因为这些功能实际上从未用于在任何决策树中拆分节点。因此，删除这些功能不会对模型结果产生影响（至少对这个特定模型来说）。

```python
import lightgbm as lgb

def identify_zero_importance_features(train, train_labels, iterations = 2):
    """
    Identify zero importance features in a training dataset based on the 
    feature importances from a gradient boosting model. 
    
    Parameters
    --------
    train : dataframe
        Training features
        
    train_labels : np.array
        Labels for training data
        
    iterations : integer, default = 2
        Number of cross validation splits to use for determining feature importances
    """
    
    # Initialize an empty array to hold feature importances
    feature_importances = np.zeros(train.shape[1])

    # Create the model with several hyperparameters
    model = lgb.LGBMClassifier(objective='binary', boosting_type = 'goss', 
                               n_estimators = 10000, class_weight = 'balanced')
    
    # Fit the model multiple times to avoid overfitting
    for i in range(iterations):

        # Split into training and validation set
        train_features, valid_features, train_y, valid_y = train_test_split(train, train_labels, 
                                                                            test_size = 0.25, 
                                                                            random_state = i)

        # Train using early stopping
        model.fit(train_features, train_y, early_stopping_rounds=100, 
                  eval_set = [(valid_features, valid_y)], 
                  eval_metric = 'auc', verbose = 200)

        # Record the feature importances
        feature_importances += model.feature_importances_ / iterations
    
    feature_importances = pd.DataFrame({'feature': list(train.columns), 
                            'importance': feature_importances}).sort_values('importance', 
                                                                            ascending = False)
    
    # Find the features with zero importance
    zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
    print('\nThere are %d features with 0.0 importance' % len(zero_features))
    
    return zero_features, feature_importances
view raw
```





剔除特征累积权重小于阈值的特征集

```python
clf = Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(dual="auto", penalty="l1"))),
  ('classification', RandomForestClassifier())
])
clf.fit(X, y)


def select_import_features(scores, thresh=0.05):
	feature_imp = pd.DataFrame(scores).sort_values('score', ascending=True)
	feature_imp['score_pct'] = feature_imp['score']/feature_imp['score'].sum()
	feature_imp['cumsum_pct'] = feature_imp['score_pct'].cumsum()
	top_features = feature_imp.query(f'cumsum_pct>={thresh}').sort_values('score', ascending=False)
  return top_features

print(top_features)
```



可以看到，我们构建的一些功能进入了前15名，这应该让我们有信心，我们所有的辛勤工作都是值得的！

```python
def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest) 
        with a column for normalized importance
        """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df
```



删除0重要功能后，我们有536个功能和另一个选择。如果我们认为我们仍然有太多的功能，我们可以开始删除重要性最小的功能。在这种情况下，我继续选择功能，因为除了gbm之外，我想测试那些在大量功能方面做得不好的模型。

我们做的最后一个功能选择步骤是只保留95%的重要性所需的功能。根据梯度增强机，342个功能足以覆盖95%的重要性。以下图显示了累积重要性与特征数量。

我们可以使用许多其他尺寸还原技术，例如[主成分分析（PCA）。](http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf?)这种方法可以有效地减少维度的数量，但它也将特征转换为低维特征空间，在那里它们没有物理表示，这意味着PCA特征无法解释。此外，PCA假设数据是正常分布的，这可能不是人类生成数据的有效假设。在笔记本中，我展示了如何使用pca，但实际上并没有将其应用于数据。

然而，我们可以使用pca进行可视化。如果我们用`TARGET`的值绘制前两个主成分，我们会得到以下图像：

这两个类别没有完全分开，只有两个主要组成部分，显然，我们需要两个以上的功能来识别将偿还贷款的客户和不偿还贷款的客户。

在继续之前，我们应该记录我们采取的功能选择步骤，以便我们记住它们以备将来使用：

1. 删除相关系数大于0.9的共线变量：*删除了583个特征*
2. 删除缺失值超过75%的列：*删除了19个功能*
3. 根据GBM删除0.0重要功能：*删除308个功能*
4. 仅保留95%功能重要性所需的功能：*删除了193个功能*

稳定性选择（Stability selection）是一种基于二次抽样和选择算法相结合较新的方法，选择算法可以是回归、SVM或其他类似的方法。

它的主要思想是在不同的数据子集和特征子集上运行特征选择算法，不断的重复，最终汇总特征选择结果。比如可以统计某个特征被认为是重要特征的频率（被选为重要特征的次数除以它所在的子集被测试的次数）。

理想情况下，重要特征的得分会接近100%。稍微弱一点的特征得分会是非0的数，而最无用的特征得分将会接近于0。

RandomizedLasso(alpha=0.025)

## 降维

常见的降维方法除了基于L1惩罚项的模型以外，另外还有主成分分析法（PCA）和线性判别分析（LDA），线性判别分析本身也是一个分类模型。PCA和LDA有很多的相似点，其本质是要将原始的样本映射到维度更低的样本空间中，但是PCA和LDA的映射目标不一样：PCA是为了让映射后的样本具有最大的发散性；而LDA是为了让映射后的样本有最好的分类性能。所以说PCA是一种无监督的降维方法，而LDA是一种有监督的降维方法。

| 方法           | 函数                       | python包                      |
| -------------- | -------------------------- | ----------------------------- |
| 主成分分析法   | PCA                        | sklearn.decomposition         |
| 线性判别分析法 | LinearDiscriminantAnalysis | sklearn.discriminant_analysis |

# 特征工程总结

main_pipeline
feature_creation
feature_selection



 

1. 单变量特征选择可以用于理解数据、数据的结构、特点，也可以用于排除不相关特征，但是它不能发现冗余特征。

2. 正则化的线性模型可用于特征理解和特征选择。相比起L1正则化，L2正则化的表现更加稳定，L2正则化对于数据的理解来说很合适。由于响应变量和特征之间往往是非线性关系，可以采用basis expansion的方式将特征转换到一个更加合适的空间当中，在此基础上再考虑运用简单的线性模型。

3. 随机森林是一种非常流行的特征选择方法，它易于使用。但它有两个主要问题：

   - 重要的特征有可能得分很低（关联特征问题）
   - 这种方法对特征变量类别多的特征越有利（偏向问题）

4. 特征选择在很多

   机器学习

   和

   数据挖掘

   场景中都是非常有用的。在使用的时候要弄清楚自己的目标是什么，然后找到哪种方法适用于自己的任务。

   - 当选择最优特征以提升模型性能的时候，可以采用交叉验证的方法来验证某种方法是否比其他方法要好。
   - 当用特征选择的方法来理解数据的时候要留心，特征选择模型的稳定性非常重要，稳定性差的模型很容易就会导致错误的结论。
   - 对数据进行二次采样然后在子集上运行特征选择算法能够有所帮助，如果在各个子集上的结果是一致的，那就可以说在这个数据集上得出来的结论是可信的，可以用这种特征选择模型的结果来理解数据。

5. 关于训练模型的特征筛选，个人建议的实施流程 :

   1. 数据预处理后，先排除取值变化很小的特征。如果机器资源充足，并且希望尽量保留所有信息，可以把阈值设置得比较高，或者只过滤离散型特征只有一个取值的特征。
   2. 如果数据量过大，计算资源不足（内存不足以使用所有数据进行训练、计算速度过慢），可以使用单特征选择法排除部分特征。这些被排除的特征并不一定完全被排除不再使用，在后续的特征构造时也可以作为原始特征使用。
   3. 如果此时特征量依然非常大，或者是如果特征比较稀疏时，可以使用PCA（主成分分析）和LDA（线性判别）等方法进行特征降维。
   4. 经过样本采样和特征预筛选后，训练样本可以用于训练模型。但是可能由于特征数量比较大而导致训练速度慢，或者想进一步筛选有效特征或排除无效特征（或噪音），我们可以使用正则化线性模型选择法、随机森林选择法或者顶层特征选择法进一步进行特征筛选。

**最后，特征筛选是为了理解数据或更好地训练模型，我们应该根据自己的目标来选择适合的方法。为了更好/更容易地训练模型而进行的特征筛选，如果计算资源充足，应尽量避免过度筛选特征，因为特征筛选很容易丢失有用的信息。如果只是为了减少无效特征的影响，为了避免过拟合，可以选择随机森林和XGBoost等集成模型来避免对特征过拟合。** 
