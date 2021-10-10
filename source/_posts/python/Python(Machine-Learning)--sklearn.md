---
ID: 2ce37637a214d0c04199cae53398ffdd
title: Python手册(Machine Learning)--sklearn
tags: [python,机器学习]
date: 2018-05-10 23:08:52
categories: [python,机器学习]
---

[**Scikit-Learn**][sk] : The most popular and widely used library for machine learning in Python.

-   分类：SVM、近邻、随机森林、逻辑回归等等。
-   回归：Lasso、岭回归等等。
-   聚类：k-均值、谱聚类等等。
-   降维：PCA、特征选择、矩阵分解等等。
-   选型：网格搜索、交叉验证、度量。
-   预处理：特征提取、标准化。

[sk]: https://sklearn.apachecn.org/

<!-- more -->

# 建模流程(The simplest sklearn workflow)

```python
X_train,y_train,x_test,y_test=getData() # 训练集和测试集
model=somemodel()# 定义模型
model.fit(X_train,y_train)# 拟合模型 
predictions=model.predict(x_test) #模型预测
model.get_params() # 获得这个模型的参数
score=model.score(x_test,y_test) # 为模型进行打分
```

**Model Fitting**
```python
# Supervised learning 
>>> lr.fit(X, y) 
>>> svc.fit(X_train, y_train)  

# Unsupervised Learning 
>>> k_means.fit(X_train) 
>>> pca_model = pca.fit_transform(X_train)
```

 **Prediction**
```python
# Supervised Estimators 
>>> y_pred = svc.predict(np.random.random((2,5))) 
>>> y_pred = lr.predict(X_test) 
>>> y_pred = knn.predict_proba(X_test)   

# Unsupervised Estimators 
>>> y_pred = k_means.predict(X_test)
```

#  数据集(Datasets)
import sklearn.datasets

## 加载示例数据集
| 加载 | 适用类型|
| ------ | ------- |
| datasets.load_iris()   | 分类   |
| datasets.load_boston() | 回归   |
| datasets.load_didits() | 分类(数字图片) |

## 创建数据集
`datasets.make_classification()`创建分类数据集
`datasets.make_regression()`创建回归数据集 
> 参数：
> n_samples：指定样本数
> n_features：指定特征数
> n_classes：指定几分类

#  回归和分类(Regression and Classification)

## 线性模型
| sklearn.linear_model| 线性模型   |
| ------- | ------- |
| LinearRegression| 普通最小二乘法 |
| Ridge   | 岭回归 |
| Lasso   | 估计稀疏系数的线性模型回归 |
| MultiTaskLasso  | 多任务 Lasso 回归   |
| LogisticRegression  | logistic 回归  |
| SGDRegressor| 随机梯度下降回归   |
| SGDClassifier   | 随机梯度下降分类   |
| ElasticNetCV| 弹性网络回归   |
| MultiTaskElasticNet | 多任务弹性网络回归 |

## 支持向量机
| sklearn.svm | 支持向量机 |
| ------- | ------- |
| SVC | 支持向量机分类 |
| SVR | 支持向量机回归 |

## 最近邻算法
| sklearn.neighbors   | 最近邻算法 |
| ------- | ------- |
| NearestNeighbors| 无监督最近邻   |
| KNeighborsClassifier| k-近邻分类|
| RadiusNeighborsClassifier   | 固定半径近邻分类   |
| KNeighborsRegressor | 最近邻回归 |
| RadiusNeighborsRegressor||
| NearestCentroid | 最近质心分类   |

## 高斯过程
| sklearn.gaussian_process| 高斯过程   |
| ------- | ------- |
| GaussianProcessRegressor| 高斯过程回归（GPR）|
| GaussianProcessClassifier   | 高斯过程分类（GPC）|
| Kernel  | 高斯过程内核   |

## 决策树
| sklearn.tree| 决策树 |
| ------- | ------- |
| DecisionTreeClassifier  | 决策树分类   |
| DecisionTreeRegressor   | 决策树回归   |

## 内核岭回归
| sklearn.kernel_ridge| 内核岭回归 |
| ------- | ------- |
| KernelRidge | 内核岭回归 |

## 等式回归
| sklearn.isotonic.IsotonicRegression | 等式回归   |
| ------- | ------- |
| IsotonicRegression |  |

## 多类多标签算法
| sklearn.multiclass  | 多类多标签算法 |
| ------- | ------- |
| multiclass |  |

## 朴素贝叶斯分类器
| sklearn.naive_bayes | 朴素贝叶斯分类器 |
| ------- | ------- |
| GaussianNB  | 高斯朴素贝叶斯 |
| MultinomialNB   | 多项分布朴素贝叶斯 |
| BernoulliNB | 伯努利朴素贝叶斯   |

## 集成方法
| sklearn.ensemble| 集成方法   |
| ------- | ------- |
| BaggingClassifier   | Bagging|
| BaggingRegressor||
| RandomForestClassifier  | 随机森林   |
| RandomForestRegressor   ||
| ExtraTreesClassifier| 极限随机树 |
| ExtraTreesRegressor ||
| AdaBoostClassifier  | AdaBoost   |
| AdaBoostRegressor   ||
| GradientBoostingClassifier  | Gradient Tree Boosting (梯度树提升)|
| GradientBoostingRegressor   |  |
| VotingClassifier| 投票分类器 |

## 神经网络
|sklearn.neural_network|神经网络|
| ------- | ------- |
|MLPClassifier|多层感知器(MLP)|
|MLPRegressor||

## 高斯混合模型
| sklearn.mixture | 高斯混合模型  |
|------|------|
| GaussianMixture | 高斯混合  |
| BayesianGaussianMixture | 变分贝叶斯高斯混合|

# 聚类(Clustering)

| sklearn.cluster | 聚类  |
|------|------|
| KMeans  | K-means聚类   |
| AffinityPropagation | AP聚类|
| MeanShift   |   |
| SpectralClustering  |   |
| AgglomerativeClustering | 层次聚类  |
| DBSCAN  |   |
| Birch   |   |
| **sklearn.cluster.bicluster**   | 双聚类|
| SpectralCoclustering|   |
| SpectralBiclustering|   |


# 降维(DimensionalityReduction)

## 矩阵分解(sklearn.decomposition)

| 主成分分析（PCA）   |   |
|------|------|
| PCA | 准确的PCA和概率解释   |
| IncrementalPCA  | 增量PCA   |
| KernelPCA   | 核 PCA|
| SparsePCA   | 稀疏主成分分析|
| MiniBatchSparsePCA  | 小批量稀疏 PCA|
| **截断奇异值分解**| |
| TruncatedSVD| |
| *词典学习：*|   |
| SparseCoder | 带有预计算词典的稀疏编码  |
| DictionaryLearning  | 通用词典学习  |
| MiniBatchDictionaryLearning | 小批量字典学习|
| **因子分析**|   |
| FactorAnalysis  | 高斯分布  |
| FastICA | 隐变量基于非高斯分布  |
| **独立成分分析（ICA）** |   |
| FastICA |   |
| **非负矩阵分解(NMF 或 NNMF)**   |   |
| NMF |   |
| **隐 Dirichlet 分配（LDA）**|   |
| LatentDirichletAllocation   |   |

## 流形学习(sklearn.manifold)

| sklearn.manifold| 流形学习，是一种非线性降维方法  |
|------|------|
| Isomap  | 等距映射（Isometric Mapping） |
| LocallyLinearEmbedding  | 局部线性嵌入  |
| SpectralEmbedding   | 谱嵌入|
| MDS | 多维尺度分析  |
| TSNE| t 分布随机邻域嵌入（t-SNE）   |


# 模型选择(ModelSelection)
import sklearn.model_selection

## 数据集拆分(train_test_split)
```python
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train,   y_test = train_test_split(X, y, test_size=None,train_size=None)
```
> 参数test_size：float(样本比例,默认0.25), int(样本量)

## 交叉验证
- 交叉验证-单指标(cross_val_score)
```python
scores=model_selection.cross_val_score(model,X,y,
  scoring=None,cv=None,
  n_jobs=1,verbose=0,fit_params=None)

scores.mean()
```
> return ndarray(包含所有的结果) 
> scoring：评分方法
> cv(Cross-validation)：cv数量
> n_jobs：并行数


- 交叉验证-多度量评估(cross_validate)：用法同cross_val_score
- 通过交叉验证获取预测(cross_val_predict)
```python
predicted=cross_val_predict(model,X,y,cv=None)
```

## 调整估计器的超参数
```python
estimator.get_params()  # 获取估计器参数

# 网格搜索
GridSearchCV(model,param_grid,scoring=None,verbose=0) 
# param_grid:dict or list
# verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。

# 随机搜索
RandomizedSearchCV()
```
> **评分参数**：[sklearn.metrics](http://sklearn.apachecn.org/cn/0.19.0/modules/model_evaluation.html) 

Examples:
```python
>>> iris =   datasets.load_iris()
>>> parameters =   {'kernel':('linear', 'rbf'), 'C':[1, 10]} #超参数
>>> svc = svm.SVC()
>>> clf =   GridSearchCV(svc, parameters)
>>> clf.fit(iris.data, iris.target)
# 属性：
>>> clf.best_params_      #最优参数
>>> clf.best_estimator_   #最优估计器
>>> clf.best_score_       #最优得分
```


## 验证曲线: 绘制得分曲线以评估模型 (validation_curve)
```python
>>> param_range=logspace(-6,0,5)

# 验证曲线
>>> train_scores,valid_scores=validation_curve(
      model, X, y,
      param_name, param_range=param_range, #参数名字和参数设定范围
      cv=None, scoring=None)

# 得分
>>> train_scores_mean=np.mean(train_scores,axis=1)
>>> valid_scores_mean=np.mean(valid_scores,axis=1)

# 可视化
>>> plt.plot(param_range,train_scores_mean,colur='r',label='training')
>>> plt.plot(param_range,valid_scores_mean,colur='g',label='Cross-validation')
>>> plt.show()
```

## 学习曲线(learning_curve)
```python
train_sizes,train_scores,valid_scores = learning_curve(model, X, y, train_sizes, cv=None, scoring=None)
```

## 模型评分[(Metric)](http://sklearn.apachecn.org/cn/0.19.0/modules/model_evaluation.html)
`import sklearn.metrics`

Function|Description
------|-------
accuracy_score|分类：准确率
log_loss|分类：损失
roc_auc_score|分类：auc值(ROC曲线下面积）
confusion_matrix |分类：混淆矩阵
mean_absolute_error|回归：平均绝对误差
mean_squared_error |回归
r2_score|回归：R^2
label_ranking_loss|排序度量
mutual_info_scor|聚类
adjusted_rand_score|聚类：调整rand指数
homogeneity_score|聚类
v_measure_score |聚类


## 模型持久化
```python
# pickle
>>> #已有建好的模型model
>>> import pickle
>>> with open('model.pickle','wb') as f:
>>>     pickle.dumps(model,f)  #保存
>>> with open('model.pickle','rb') as f:
>>>     clf = pickle.loads(f)  #读取

# joblib
>>> from  sklearn.externals import joblib
>>> joblib.dump(model, 'filename.pkl')  # 保存
>>> model = joblib.load('filename.pkl')  # 导入
```


#  特征选择(sklearn.feature_selection)

|sklearn.feature_selection|  |
|-------|-------|
|VarianceThreshold| 移除低方差特征，单变量特征选择|
| SelectKBest  | 移除那些除了评分最高的 K 个特征之外的所有特征|
| SelectPercentile | 移除除了用户指定的最高得分百分比之外的所有特征   |
| GenericUnivariateSelect  | 允许使用可配置方法来进行单变量特征选择。它允许超参数搜索评估器来选择最好的单变量特征 |
| RFECV| 递归式特征消除   |

> 检验：
> 对于回归: f_regression , mutual_info_regression
> 对于分类: chi2 , f_classif , mutual_info_classif


#  预处理(Pre-Processing)
import sklearn.preprocessing

## 标准化
- 标准化
```python
X_scaled =   preprocessing.scale(X_train)
```
- 将特征缩放至特定范围内
```python
# MinMaxScaler
min_max_scaler =   preprocessing.MinMaxScaler()
X_train_minmax =   min_max_scaler.fit_transform(X_train)
X_test_minmax =   min_max_scaler.transform(X_test)
# X_std   =(X-X.min)/(X.max-X.min)
# X_scaled = X_std * (max -   min) + min

# MaxAbsScaler
max_abs_scaler   = preprocessing.MaxAbsScaler()
X_train_maxabs =   max_abs_scaler.fit_transform(X_train)
X_test_maxabs =   max_abs_scaler.transform(X_test)
# X_std =X/X.max
```
- 缩放有离群值的数据
```python
RobustScaler =   preprocessing.RobustScaler()
X_train_RobustScaler =   RobustScaler.fit_transform(X_train)
X_test_RobustScaler =   RobustScaler.transform(X_test)
```

## 非线性转换(无参数转换)
```python
quantile_transformer =   preprocessing.QuantileTransformer(random_state=0)
X_train_trans =   quantile_transformer.fit_transform(X_train)
X_test_trans =   quantile_transformer.transform(X_test)
```

## 归一化
```python
X_normalized =   preprocessing.normalize(X, norm='l2')
```

## 二值化
```python
binarizer =   preprocessing.Binarizer(threshold=0.0)
binarizer.transform(X)
```

## 分类特征编码
```python
enc =   preprocessing.OneHotEncoder(n_values='auto', categorical_features='all')
enc.fit(X_train)
enc.transform(X_test).toarray()
```

## 缺失值插补
```python
imp =   preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X_train)
imp.transform(X_test)
```
> 参数：strategy :{'mean','median','most_frequent'}

## 生成多项式特征
```python
X=[X1,X2]
poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=False)
poly.fit_transform(X)
```
> 参数：
> degree=2：X   的特征从[X1,X2]转换为[1,X1,X2,X1^2,X1X2,X2^2]
> interaction_only：只有交互项[1,X1,X2,X1X2]





# 其他

## 协方差估计(sklearn.covariance)
`empirical_covariance`  经验协方差

## 特征提取(sklearn.feature_extraction)
可用于提取符合机器学习算法支持的特征，比如文本和图片。

##  概率校准(sklearn.calibration)
CalibratedClassifierCV



