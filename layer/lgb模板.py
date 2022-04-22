import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from scipy.misc import derivative
from sklearn.metrics import f1_score
from datetime import datetime

pd.set_option("display.max_columns",100)

root_dir="../output/WTB_GD_A/toUserA/"
train_info_path="model_online_car_base_info_train.csv"
train_travel_path="model_online_car_hailing_travel_train.csv"
test_info_path="model_online_car_base_info_A_nolabel.csv"
test_travel_path="model_online_car_hailing_travel_A_version.csv"

train_info=pd.read_csv(root_dir+train_info_path)
test_info=pd.read_csv(root_dir+test_info_path)

train_travel=pd.read_csv(root_dir+train_travel_path)
test_travel=pd.read_csv(root_dir+test_travel_path)

data_travel=pd.concat([train_travel,test_travel],axis=0)

def timecg(x):
    x=str(x)
    return x[:4]+"-"+x[4:6]+"-"+x[6:8]+" "+x[8:10]+":"+x[10:12]+":"+x[12:]

data_travel["base_stastion_in_time_change"]=data_travel["base_stastion_in_time"].apply(timecg)
data_travel["base_stastion_out_time_change"]=data_travel["base_stastion_out_time"].apply(timecg)
data_travel.base_stastion_in_time_change=pd.to_datetime(data_travel.base_stastion_in_time_change)
data_travel.base_stastion_out_time_change=pd.to_datetime(data_travel.base_stastion_out_time_change)

data_travel["time_delta"]=data_travel.base_stastion_out_time_change-data_travel.base_stastion_in_time_change
data_travel["time_delta"]=np.array(data_travel["time_delta"].values//(10**9),dtype=np.int)

coordinate_cnt_feats=['id','base_stastion_lon','base_stastion_lat']
data_travel_coordinate_cnt=data_travel[coordinate_cnt_feats].groupby('id')['base_stastion_lon','base_stastion_lat'].count().reset_index()

def getmax(df1,df2):
    res=[]
    for i,j in zip(df1,df2):
        res.append(max(i,j))
    return res

data_travel_coordinate_cnt['coordinate_cnt']=getmax(data_travel_coordinate_cnt.base_stastion_lon.values,data_travel_coordinate_cnt.base_stastion_lat.values)
data_travel_coordinate_cnt=data_travel_coordinate_cnt[['id','coordinate_cnt']]

travel_coordinate_day=data_travel.groupby(['id','day'])['base_stastion_lon','base_stastion_lat'].count().reset_index()
travel_time_day=data_travel.groupby(['id','day'])['time_delta'].sum().reset_index()

travel_coordinate_day['day_coordinate_cnt']=getmax(travel_coordinate_day.base_stastion_lon.values,travel_coordinate_day.base_stastion_lat.values)

travel_coordinate_day_sum=travel_coordinate_day.groupby('id')['day_coordinate_cnt'].sum().reset_index()
travel_coordinate_day_sum.columns=["id","day_coordinate_cnt_sum"]

travel_coordinate_day_mean=travel_coordinate_day.groupby('id')['day_coordinate_cnt'].mean().reset_index()
travel_coordinate_day_mean.columns=["id","day_coordinate_cnt_mean"]

travel_coordinate_day_std=travel_coordinate_day.groupby('id')['day_coordinate_cnt'].std().reset_index()
travel_coordinate_day_std.columns=["id","day_coordinate_cnt_std"]

travel_coordinate_day_min=travel_coordinate_day.groupby('id')['day_coordinate_cnt'].min().reset_index()
travel_coordinate_day_min.columns=["id","day_coordinate_cnt_min"]

travel_coordinate_day_max=travel_coordinate_day.groupby('id')['day_coordinate_cnt'].max().reset_index()
travel_coordinate_day_max.columns=["id","day_coordinate_cnt_max"]

test_info["label"]=-1
data=pd.concat([train_info,test_info],axis=0)
data=data.fillna(-99)

select_num_cols=["arpu","flux","mou","call_cnt","called_cnt","call_per_cnt","map_app_flux","map_app_cnt","video_app_flux","video_app_cnt","music_app_flux","music_app_cnt"]
travel_cols=["base_stastion_lon","base_stastion_lat","time_delta"]

data_sum=data.groupby("id")[select_num_cols].sum().reset_index()
data_sum.columns=["id"]+[i+"_sum" for i in select_num_cols]

data_mean=data.groupby("id")[select_num_cols].mean().reset_index()
data_mean.columns=["id"]+[i+"_mean" for i in select_num_cols]

data_std=data.groupby("id")[select_num_cols].std().reset_index()
data_std.columns=["id"]+[i+"_std" for i in select_num_cols]

data_max=data.groupby("id")[select_num_cols].max().reset_index()
data_max.columns=["id"]+[i+"_max" for i in select_num_cols]

data_min=data.groupby("id")[select_num_cols].min().reset_index()
data_min.columns=["id"]+[i+"_min" for i in select_num_cols]

travel_sum=data_travel.groupby("id")['time_delta'].sum().reset_index()
travel_sum.columns=["id","time_delta_sum"]

travel_mean=data_travel.groupby("id")[travel_cols].mean().reset_index()
travel_mean.columns=["id"]+[i+"_mean" for i in travel_cols]

travel_std=data_travel.groupby("id")[travel_cols].std().reset_index()
travel_std.columns=["id"]+[i+"_std" for i in travel_cols]

travel_max=data_travel.groupby("id")[travel_cols].max().reset_index()
travel_max.columns=["id"]+[i+"_max" for i in travel_cols]

travel_min=data_travel.groupby("id")[travel_cols].min().reset_index()
travel_min.columns=["id"]+[i+"_min" for i in travel_cols]

data_dum=data.groupby("id")[["age","sex","innet_dur","id_card_usr_cnt","terminal","terminal_brand","label"]].max().reset_index()

le=LabelEncoder()
le=le.fit(data_dum.terminal_brand.map(lambda x:"null" if x==-99 else x))
data_dum.terminal_brand=le.transform(data_dum.terminal_brand.map(lambda x:"null" if x==-99 else x))

tcs=data_dum.terminal_brand.value_counts().to_dict()
data_dum["terminal_brand_cn"]=data_dum.terminal_brand.map(tcs)

data_dum=data_dum.merge(data_sum,how="left",on="id")
data_dum=data_dum.merge(data_mean,how="left",on="id")
data_dum=data_dum.merge(data_std,how="left",on="id")
data_dum=data_dum.merge(data_max,how="left",on="id")
data_dum=data_dum.merge(data_min,how="left",on="id")

data_dum=data_dum.merge(travel_sum,how="left",on="id")
data_dum=data_dum.merge(travel_mean,how="left",on="id")
data_dum=data_dum.merge(travel_std,how="left",on="id")
data_dum=data_dum.merge(travel_max,how="left",on="id")
data_dum=data_dum.merge(travel_min,how="left",on="id")

data_dum=data_dum.merge(travel_coordinate_day_sum,how="left",on="id")
data_dum=data_dum.merge(travel_coordinate_day_mean,how="left",on="id")
data_dum=data_dum.merge(travel_coordinate_day_std,how="left",on="id")
data_dum=data_dum.merge(travel_coordinate_day_max,how="left",on="id")
data_dum=data_dum.merge(travel_coordinate_day_min,how="left",on="id")

data_dum["id_len"]=data_dum.id.map(lambda x:len(str(x)))
data_age_cnt=data_dum.groupby("age")[["id"]].count()
data_dum["age_id_cnt"]=data_dum["age"].map(lambda x:data_age_cnt.to_dict()["id"][x])

cat_cols=["age","sex","id_card_usr_cnt","terminal","terminal_brand"]
for col in cat_cols:
    data_dum[col]=data_dum[col].astype("category")

train_data=data_dum[data_dum.label!=-1].reset_index(drop=True)
test_data=data_dum[data_dum.label==-1].reset_index(drop=True)

y=train_data.label.map(lambda x:1 if x>0 else 0).values
feats=[i for i in train_data.columns if i not in ["id","label"]]

params={
    "boosting_type":"gbdt",
    "force_row_wise":True,
    "objective":"binary",
    "metric":["auc","bianry_logloss"],
    "subsample":0.8,
    "subsample_freq":3,
    "num_leaves":45,
    "max_depth":10,
    "learning_rate":0.1,
    "min_data_in_leaf":100,
    "colsample_bytree":0.8,
    "min_child_samples":20,
    "min_child_weight":0.001,
    "min_split_gain":0.0,
    "n_jobs":-1,
    "seed":2022,
}

skf = StratifiedKFold(n_splits=2, random_state=2022)
oof_lgb = np.zeros(train_data.shape[0])
test_pred = np.zeros(test_data.shape[0])
for i, (tr, va) in enumerate(skf.split(train_data, y)):
    print("===========================fold:", i + 1, "training===========================")
    x_train, x_valid, y_train, y_valid = train_data[feats].iloc[tr], train_data[feats].iloc[va], y[tr], y[va]
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_valid = lgb.Dataset(x_valid, y_valid, reference=lgb_train)
    gbm = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_valid],
        valid_names=["valid"],
        num_boost_round=500,
        verbose_eval=10,
        early_stopping_rounds=50)
    oof_lgb[va] = gbm.predict(x_valid, num_iteration=gbm.best_iteration)
    test_pred += gbm.predict(test_data[feats], num_iteration=gbm.best_iteration) / 2

probas = gbm.predict(train_data[feats], num_iteration=gbm.best_iteration)
thresh = 0.3
preds = []
for i in probas:
    if i > thresh:
        preds.append(1)
    else:
        preds.append(0)
print(f1_score(y, preds))

res = test_data[["id"]]
res["predprob"] = test_pred
res["predtype"] = res["predprob"].map(lambda x: 1 if x > 0.3 else 0)

res.to_csv(root_dir + "submits/submita0421_2fold.csv", index=False)
print("文件保存完毕")