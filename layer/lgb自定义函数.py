import lightgbm as lgb
from scipy.misc import derivative

#数据处理

#focal loss
#LGB+Focal Loss 其中alpha：为不能让容易分类类别的损失函数太小, 默认值0.25；
#gamma：更加关注困难样本 即关注y=1的样本 默认值2
def focal_loss( y_pred,dtrain, alpha=0.25, gamma=2):
    label = dtrain.get_label()
    a,g = alpha, gamma
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
    partial_fl = lambda x: fl(x, label)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess

#自定义f1评价指标
def f1_score(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(4, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', score_vali, True


params = {
    "learning_rate": 0.1,
    "boosting": 'gbdt',  
    "lambda_l2": 0.1,
    "max_depth": -1,
    "num_leaves": 128,
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "metric": None,
    "objective": "multiclass",
    "num_class": 4,
    "nthread": 10,
    "verbose": -1,
}


gbm = lgb.train(params,
          	train_set=train_matrix, 
          	valid_sets=valid_matrix, 
                num_boost_round=2000, 
          	verbose_eval=100, 
          	early_stopping_rounds=200,
                fobj=focal_loss,
                feval=f1_score,)

#fobj：自定义损失函数
#feval：自定义评价指标


#预测
#计算验证集指标
