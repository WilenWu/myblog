---
title: Python手册(Machine Learning)--Machine Learning Tutorial
tags:
  - Python
  - 机器学习
categories:
  - Python
  - Machine Learning
cover: /img/FeatureEngine.png
top_img: /img/python-top-img.svg
abbrlink: 
date: 
description:
---

# Step 1: Imports and Configuration

```python
import pandas as pd
import numpy as np
import copy
import json
import pickle
import lightgbm as lgb
import optuna

from sklearn.metrics import roc_curve, roc_auc_score, recall_score, accuracy_score, fbeta_score
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from focal_loss import BinaryFocalLoss # self-define loss function

pd.set_option('display.max_columns',None) 
pd.set_option('display.max_rows',None) 
pd.set_option('maxcolwidth',2000)
pd.set_option('display.float_format', lambda x: '%.5f' %x)
```


先定义一个计时器，方便后续评估性能

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
		print(f"{func.__name__} start running...")
	    click = time.time()
	    result = func(*args, **kwargs)
	    delta = strfdelta(time.time() - click, "{:.0f} hours {:.0f} minutes {:.0f} seconds")
	    print(f"{func.__name__} end. And cost time {delta}")
	    return result
	return wrapper
```
# Step 2: Load the datasets


```python
print('Loading data...')
path = '~/opt/datasets/'
features = pd.read_excel(path + 'top_features.xlsx', index=0).index.to_list() 

data = pd.read_csv(path+'requared.csv', columns=features) 
data.columns
data.head()

# Specific feature names and categorical features
#categorical_features = 
# numeric_cols
# ID_col
target = "label"

X = data.loc[:, features]
y = data[target]
```

留30%作为模型的验证集

```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("X_train shape: {}x{}".format(*X_train.shape))
print('train:', y_train.value_counts(), sep='\n') 
print('test:', y_test.value_counts(), sep='\n')
```

# Step 3: Data preprocessing

## 样本不平衡


lgb详细超参数测试下focalLoss和设置unbalance+auc 和auc

```python
# Check if the data is unbalanced
```

样本不平衡：权重调整法、重采样、SMOTE方法、切分训练法


model 1: use default parameter
model 2: use smot
```python
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = sm.fit_resample(X_train.fillna(0), y_train)
print('bad rate is: ',y_train_balanced.mean())

model2 =AdaBoostClassifier(random_state=42)
model2.fit(X_train_balanced,y_train_balanced)
```

model 3: easy ensemble
```python
from imblearn.ensemble import EasyEnsembleClassifier
model3 = EasyEnsembleClassifier(n_estimators=20, random_state=42, base_estimator=AdaBoostClassifier(random_state=42))
model3.fit(X_train.fillna(0),y_train)
```


model4: FocalLoss

```python
fl = FocalLoss(alpha=0.9, gamma=0.05)
fit = lgb.Dataset(X_fit, y_fit, init_score=np.full_like(y_fit, fl.init_score(y_fit), dtype=float))
val = lgb.Dataset(X_val, y_val, reference=fit,init_score=np.full_like(y_val, fl.init_score(y_val), dtype=float))

model = lgb.train(
    params={
        'learning_rate': 0.01,
        'seed': 2021
    },
    train_set=fit,
    num_boost_round=10000,
    valid_sets=(fit, val),
    valid_names=('fit', 'val'),
    early_stopping_rounds=20,
    verbose_eval=100,
    fobj=fl.lgb_obj,
    feval=fl.lgb_eval
)

y_pred = special.expit(fl.init_score(y_fit) + model.predict(X_test))

print()
print(f"Test's ROC AUC: {roc_auc_score(y_test, y_pred):.5f}")
print(f"Test's logloss: {log_loss(y_test, y_pred):.5f}")
```

本文准备选用LightGBM分类器，需要创建 lightgbm 原生数据集
```python
# Create Dataset object for lightgbm
dtrain = lgb.Dataset(
    X_train, label=y_train, 
    feature_name=feature_name, 
    categorical_feature=categorical_feature,
    free_raw_data=False
    )

# In LightGBM, the validation data should be aligned with training data.
# if you want to re-use data, remember to set free_raw_data=False
dtest = lgb.Dataset(
    X_test, label=y_test, 
    reference=dtrain, 
    feature_name=feature_name, 
    categorical_feature=categorical_feature,
    free_raw_data=False
    )
```

# Step 4: Hyperparameter Tuning

超参数调优算法主要有网格搜索(Grid Search)，随机搜索(Randomized Search)和贝叶斯优化(Bayesian Optimization)，本文采用贝叶斯优化。

超参数和目标函数设置

Here we use Optuna
```python
# LightGBM can use a dictionary to set Parameters.
# Booster parameters:
def objective(trial, X, y):
params = dict(
    boosting_type='gbdt',
    objective='binary',
    metric='auc',
    nthread=4,
    device='cpu',
    n_trees=trial.suggest_int("n_trees", 100, 500),
    learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True), 
    max_depth=trial.suggest_int("max_depth", 2, 10), 
    min_child_weight=trial.suggest_int("min_child_weight", 1, 10), 
    feature_fraction=trial.suggest_float("feature_fraction", 0.2, 1.0), 
    bagging_fraction=trial.suggest_float("bagging_fraction", 0.2, 1.0),  
    bagging_freq=5,
    lambda_l1=trial.suggest_float("lambda_l1", 1e-4, 1e2, log=True),  
    lambda_l2=trial.suggest_float("lambda_l2", 1e-4, 1e2, log=True)  
)
evals_result={} # to record eval results for plotting
callbacks=[
    lgb.log_evaluation(period=5), 
    lgb.early_stopping(stopping_rounds=5),
    lgb.record_evaluation(evals_result)
    ]
    clf = LGBClassfier(**params)
    score = cross_val_score(estimator, X, y, scoring='roc_auc', cv=5)
    return scores.mean(), scores.std()
```



超参数调优

```python
@timer
def Bayesian_optimizer(objective, X, y):
	study = optuna.create_study(direction="minimize")
	study.optimize(lambda X, y: objective(trial, X, y), n_trials=20)
	return study

clf = 
scores = cross_val_score(clf, X, y, scoring='roc_auc', cv=5)
print(f"AUC: {scores.mean():.2f} (+/- {scores.std():.2f}) [{label}]" )
```



调优过程曲线，带宽为标准差

# Step 5: Training

```python
print('Starting training...')
bst = lgb.train(
    best_params, 
    dtrain, 
    valid_sets=[dtrain, dtest],
    num_boost_round=50,
    callbacks=callbacks
    )

# Training with 5-fold CV:
bst = lgb.cv(params, dtrain, num_boost_round=20, nfold=5)



# Plotting metrics recorded during training
ax = lgb.plot_metric(evals_result, metric='auc')
plt.show()
```


```python
X_train, X_test = create_features(df_train, df_test)
y_train = df_train.loc[:, "SalePrice"]

xgb = XGBRegressor(**xgb_params)
# XGB minimizes MSE, but competition loss is RMSLE
# So, we need to log-transform y to train and exp-transform the predictions
xgb.fit(X_train, np.log(y))
predictions = np.exp(xgb.predict(X_test))

output = pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions})
output.to_csv('my_submission.csv', index=False)
```





# Step 6: Evaluating

## 模型得分
```python
def get_adjusted_prediction(y_score, threshold=0.5):
    y_pred = y_score.copy()
    y_pred[y_score>=threshold] = 1
    y_pred[y_score< threshold] = 0
    return y_pred

def classification_report(model, **valid_sets):
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.metrics import recall_score, accuracy_score, fbeta_score, precision_score
    report = {}
    for key, dataset in valid_sets.items():
        y_true = dataset.get_label()
        y_score = model.predict(dataset.get_data()) 
        y_pred = np.argmax(y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score) 
        idx = (tpr - fpr).argmax()
        threshold = thresholds[idx]
        adjusted_y_pred = get_adjusted_prediction(y_score, threshold)
        stats = {
            'y_true': y_true,
            'y_score': y_score,
            'fpr': fpr,
            'tpr': tpr, 
            'threshold': threshold,
            'ks': (tpr-fpr).max(),
            'auc': roc_auc_score(y_true, y_score),
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy_score': None,
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1-score': fbeta_score(y_true, y_pred, beta=1),
            'adjusted_accuracy': accuracy_score(y_true, adjusted_y_pred)
            }
        report[key] = stats
    return report

report  = classification_report(bst, train=dtrain, test=dtest)

# the model performance
for label, stats in report.items():
    stats['label'] = label
    s = "[{label}] auc: {auc:.2f}, " + \
        "accuracy: {accuracy:.2f}, " + \
        "recall: {recall:.2f}, " + \
        "adjusted_accuracy(threshold = {threshold:.4f}): {adjusted_accuracy:.2f} " 
    print(s.format(**stats))
```

## ROC曲线
```python
# Plot ROC curve
def plot_roc_curve(fprs, tprs, labels):
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate') 
	plt.ylabel('True Positive Rate') 
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # Plotting ROC and computing AUC scores
    for fpr, tpr in zip(fprs, tprs, labels):
    	auc = metrics.auc(fpr, tpr)
    	plt.plot(fpr, tpr, 'b', label = f"{label} ROC(auc={auc:.2f})")
    plt.legend(loc = 'lower right')
```


## 模型稳定性

PSI(Population Stability Index)指标反映了实际分布(actual)与预期分布(expected)的差异。在建模中，我们常用来筛选特征变量、评估模型稳定性。其中，在建模时通常以训练样本(In the Sample, INS)作为预期分布，而验证样本在各分数段的分布通常作为实际分布。验证样本一般包括样本外(Out of Sample, OOS)和跨时间样本(Out of Time, OOT)。 

> 风控模型常用PSI衡量模型的稳定性。

```python
def calc_psi(expected, actual, bins=10):

'''Calculate the PSI (Population Stability Index) for two vectors.



Args:

expected: array-like, represents the expected distribution.

actual: array-like, represents the actual distribution.

bins: int, the number of bins to use in the histogram.



Returns:

float, the PSI value.

'''

# Calculate the bin boundaries

breakpoints = np.linspace(0, 1, bins+1)



# Calculate the expected frequencies in each bin

expected_freq = np.histogram(expected, breakpoints)[0]

expected_freq = expected_freq / float(sum(expected_freq))



# Calculate the actual frequencies in each bin

actual_freq = np.histogram(actual, breakpoints)[0]



# score distribution
n = 10
expected_score = report['train']['y_score']
cuts, bins = pd.cut(expected_score, n, labels=False, retbins=True, duplicates='drop')
expected_dist = cuts.value_counts().sort_index()
print(expected_dist)
print(bins)

# 保存期望分布
expected_dist.to_csv('expected_score_distribution.csv')
np.savetxt('expected_score_bins.txt', bins)

def get_psi(actual, expected_dist, bins): 
    actual = pd.Series(actual) 
    bins  = [-np.inf] + bins.tolist()[1:-1] + [np.inf]
    actual_cuts = pd.cut(actual, bins, labels=False, duplicates='drop')
    actual_dist = actual_cuts.value_counts()
    psi_df = pd.DataFrame({'actual_dist': actual_dist, 'expected_dist': expected_dist}) 
    psi_df[['actual_ratio', 'expected_ratio']] = psi_df/psi_df.sum()
    psi_df['psi'] = (psi_df['actual_ratio'] - psi_df['expected_ratio']) * \
                     np.log(psi_df['actual_ratio'] / psi_df['expected_ratio'])
    psi = psi_df['psi'].sum()
    return psi_df, psi

expected_dist = pd.read('expected_score_distribution.csv')
expected_bins = np.loadtxt('expected_score_bins.txt')
actual_score = report['test']['y_score']

psi_df, psi = get_psi(actual_score, expected_dist, expected_bins)
print(f"psi: {psi}")

# 绘制实际分布与预期分布曲线
# 改用for 循环画图
fig=plt.figure(figsize=(15,5)) 
ax1=fig.add_subplot(121)
sns.kdeplot(y_prob_train,ax=ax1) 
sns.kdeplot(y_prob_test,ax=ax1) ax1.set_xscale('log')
ax2=fig.add_subplot(122)
sns.kdeplot(y_prob_train,ax=ax2) sns.kdeplot(y_prob_test,ax=ax2) 
plt.show()

## 正负样本得分分布，改用for 循环画图
ro = y_prob_train[y_train ==0] 
r1= y_prob_train[y_train==1]
fig = plt.figure(facecolor ='white',figsize =(20,5)) 
ax1 =fig.add_subplot(1,2,1)
sns.distplot(r0,kde_kws = {"label":"y"},color = 'r',ax=ax1)
sns.distplot(r1,kde_kws = {"label":"pred"},color = 'g',ax=ax1)
ax1.set_title(label ='Frequency',loc ='center') 
ax2=fig.add_subplot(1,2,2)
sns.kdeplot(r0,cumulative=True,label ="y",color ='r',ax=ax2) 
sns.kdeplot(r1,cumulative=True,label ="pred",color='g',ax=ax2) 
ax2.set_title(label='cumulative',loc ='center')
```
# Step 7: Show feature importance
```python
feature_imp = pd.DataFrame(bst.feature_importance(), index=bst.feature_name, columns=['score'])
feature_imp.sort_values('score', ascending=False, inplace=True)
feature_imp.to_excel(path+'feature_importance.xlsx')

# Plotting feature importances
ax = lgb.plot_importance(bst, max_num_features=20)
plt.show()

# 观察重点特征的分布 
for col in imp_top.index:
    df=pd.DataFrame({'x':X_train[col],'label':y_train}) 
    df[col] = pd.qcut(df['x'],6,duplicates='drop')
    df_pivot=df.pivot_table(index=col,columns='label',values='label',aggfunc='count')
    df_pivot.plot(kind='barh')

# 特征分箱后正负样本得分的箱线图
```
# Step 8: Visualize the model
```python
# Plotting split value histogram
ax = lgb.plot_split_value_histogram(bst, feature='x26', bins='auto')
plt.show()

# Plotting 54th tree (one tree use categorical feature to split)
ax = lgb.plot_tree(bst, tree_index=53, figsize=(15, 15), show_info=['split_gain'])
plt.show()

# Plotting 54th tree with graphviz
graph = lgb.create_tree_digraph(bst, tree_index=53, name='Tree54')
graph.render(view=True)
```

# Step 9: Model persistence
```python
# Save model to file
print('Saving model...')
bst.save_model('model.txt')  
```
# Step 10: Predict
```python
# Perform predictions
# If early stopping is enabled during training, you can get predictions from the best iteration with bst.best_iteration:
y_pred = bst.predict(X_test)

# Load a saved model to predict 
print('Loading model to predict...')
bst = lgb.Booster(model_file='model.txt')
y_pred = bst.predict(X_test)

# Save predictions
```


# Ensembles

```python
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# Define a cross validation strategy
#We use the cross_val_score function of Sklearn. However this function has not a shuffle attribut, we add then one line of code, in order to shuffle the dataset prior to cross-validation
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

#Base models
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

```

有时候模型集成可以取得不错的效果。常用的模型集成包括：

- Votting：简单投票或加权平均
- Stacking：简单来说就是学习各个基本模型的预测值来预测最终的结果

我们初步选用 Stacking 集成学习器，采用 LogisticRegression、RandomForest、NaiveBayes、LightGBM、SVM作为第一层基分类器，采用LightGBM作为第二层基分类器。

Then we choose many base models (mostly sklearn based models + sklearn API of DMLC's XGBoost and Microsoft's LightGBM), cross-validate them on the data before stacking/ensembling them. 
## 基分类器

无序分类（unordered）特征原始编码对于树集成模型（tree-ensemble like XGBoost）是可用的，但对于线性回归模型（like Lasso or LogisticRegression）则必须使用one-hot重编码。因此，接下来对于不同的模型采用不同的特征编码。

## 创建数据集

```python
def create_dataset(df):
	return X, y, feature_names_out

X_train, y_train, _ = create_dataset(df_train)
```

## 创建优化器

选用auc作为评估指标

```python
def score_dataset(X, y, model=XGBRegressor()):
    # The `cat.codes` attribute holds the category levels.
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    log_y = np.log(y)
    score = cross_val_score(
        model, X, log_y, cv=5, scoring="neg_root_mean_squared_error",
    ).mean()
    return -1 * score
```

```python
# Setting hyperparameters & Objective
from sklearn.ensemble import StackingClassfier

class Optimizer:
  
  def __init__(self, label, X, y):
  	"""
  	Params:
  	label: Logistic Regression, Random Forest, Naive Bayes, Support Vector Machine, LightGBM
  	"""
    self.label = label
    self.X = X
    self.y = y
  
  def objective(self, trial):
    self._search_space = {
      'Logistic Regression': dict(
          w=
      ), 
      'Random Forest': dict(),
      'Naive Bayes': dict(),
      'Support Vector Machine': dict(),
      'LightGBM': dict()
    }
 
    self.meta_estimators = {}
    estimators = [LogisticRegression, RandomForestClassfier, NaiveBayesClassfier, SVM, LightGBMClassfier]
    for (label, params), estimator in zip(self._search_space.items(), estimators):
    	self.meta_estimators[label] = estimator(**params)
    
    self.space = self._search_space.get(self.label, self._search_space)
	if self.label in self._search_space.keys():
		self.estimator = self.meta_estimators[self.label]
	elif self.label == 'Stacking':
		self.estimator = StackingRegressor(
   			 	# The `estimators` parameter corresponds to the list of the estimators which are stacked.
    			estimators=self.meta_estimators,
    			# The final_estimator will use the predictions of the estimators as input
    			final_estimator=GradientBoostingClassfier(**gbc_params)
		scores = cross_val_scores(self.estimator, self.X, self.y, cv=5)
		return scores.mean(), scores.std()
	
    @timer
    def optimize(self, *args, **kwargs):
        study = optuna.create_study(direction=["max", "minimize"])
        study.optimize(self.objective, *args, **kwargs)
        return study
```


## 超参数优化

并行执行贝叶斯优化
```python
# Creating a pipeline & Hyperparameter tuning
import concurrent.futures

start_time = time.time()
optimizers = {}
labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Support Vector Machine', 'LightGBM']
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:  
    task_to_label = {}
    for label in labels:
        opt = Optimizer(label, df)
        opt.optimize.__name__ = label
        task_to_label[executor.submit(opt.optimize, n_trials=20)] = label 
    
    for task in concurrent.futures.as_completed(task_to_label):
        label = task_to_label[task]
        optimizers[label] = task.result()

print ("Thread pool execution in {:.0f} seconds".format(time.time() - start_time)

for label, study in optimizers.items():
	print(f"{label}'s best score: {study.best_score}")
```

## 模型训练

选择一个合适的模型训练
```python
best_model_label = 'LightGBM'
best_params = optimizers[best_model_label].best_params
best_model = LightGBMClassfier(**best_params)
best_model.fit(X_train, y_train)
best_model.score(X_test, y_test)
```

