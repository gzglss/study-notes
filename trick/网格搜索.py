#以逻辑回归作为训练模型
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

mll_scorer = metrics.make_scorer(multiclass_logloss,
                       greater_is_better=False, needs_proba=True)

svd = TruncatedSVD()

# Standard Scaler初始化
scl = preprocessing.StandardScaler()

# Logistic Regression
lr_model = LogisticRegression()

# 创建pipeline
clf = pipeline.Pipeline([('svd', svd),
                         ('scl', scl),
                         ('lr', lr_model)])

param_grid = {'svd__n_components' : [120, 180],
              'lr__C': [0.1, 1.0, 10],
              'lr__penalty': ['l1', 'l2']}

model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)
model.fit(xtrain_tfv, ytrain)  #为了减少计算量，这里我们仅使用xtrain
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))








#以朴素贝叶斯作为训练模型
nb_model = MultinomialNB()

clf = pipeline.Pipeline([('nb', nb_model)])

param_grid = {'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)
model.fit(xtrain_tfv, ytrain)  # 为了减少计算量，这里我们仅使用xtrain
print("Best score: %0.3f" % model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))