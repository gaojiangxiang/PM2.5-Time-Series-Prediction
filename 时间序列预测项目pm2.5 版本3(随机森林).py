"""
时间序列预测项目：

数据为北京市2014年11月至12月每天每小时的pm2.5值 共730个数据

用2014年11月30个小时的pm2.5数值 温度 气压 风速 过去24小时平均污染水平 最严重污染水平 波动程度 预测以后每小时的pm2.5值

随机森林回归算法预测

均方误差: 1390.8232893165352
平均绝对误差: 22.286236135975628
R2= 0.9266101285875957 （相较v2降低了0.031）

在当前特征工程和参数的设置下，线性回归表现优于随机森林 数据中可能以线性关系为主
"""

import sklearn
import pandas as pd #导入数据 处理
import random #导入随机数据
import matplotlib.pyplot as plt #导入图形，数据可视化
import numpy as np #导入数据，数组形式
from sklearn.ensemble import RandomForestRegressor #导入随机森林包
from sklearn.model_selection import train_test_split #导入划分数据集的包
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score #导入模型评估公式 均方误差 平均绝对误差

#----------------------1.获取数据集合， 处理数据---------------
df = pd.read_csv("air_quality.csv") #读取数据
df1 = df.iloc[1:731, 1:6] #只取年月日时 以及pm2.5的数据 即数据2-6列 以及2014年整个十一月的数据 即2-731行
df2 = df[ ["TEMP","PRES","Iws"] ] #取天气特征 temp温度 pres大气压 iws累计风速
df3 = df2.iloc[1:731, :] #取天气特征的2014年整个十一月的数据 即2-731行
df = pd.concat([df1,df3], axis = 1) #合并列对象（添加列特征） axis=1方向

print( df.to_string(index=False) ) # 去掉前面的索引
print(df.shape) #打印df的形状

#查看是否有缺失值 处理缺失值
print("pm2.5缺失值数量:", df["pm2.5"].isnull().sum()) #显示有14个缺失值 2014-11-20-21至2014-11-21-8
df["pm2.5"] = df["pm2.5"].interpolate() #在缺失值的位置进行插值，数据预处理方法
print("pm2.5缺失值数量:", df["pm2.5"].isnull().sum()) #显示有0个缺失值

print("TEMP缺失值数量:", df["TEMP"].isnull().sum()) #显示有0个缺失值，所以不用补全
print("PRES缺失值数量:", df["PRES"].isnull().sum()) #显示有0个缺失值，所以不用补全
print("Iws缺失值数量:", df["Iws"].isnull().sum()) #显示有0个缺失值，所以不用补全

#特征工程处理 生成新的特征
df["pm2.5_mean_24"] = df["pm2.5"].shift(1).rolling(24).mean() #过去24小时平均污染水平
print( df["pm2.5_mean_24"] )
df["pm2.5_max_24"] = df["pm2.5"].shift(1).rolling(24).max() #过去24小时最严重污染水平
print( df["pm2.5_max_24"] )
df["pm2.5_std_24"] = df["pm2.5"].shift(1).rolling(24).std() #过去24小时波动程度
print( df["pm2.5_std_24"] )

#只写rolling（24），计算的是77-100小时，包含pm2.5[100]，模型已经知道答案 R2=1.0
#加了shift（1） 所有数据向下移动1行 此时第一百行计算的是76-99小时，模型不知道答案 在做预测
#为了避免数据泄露

df = df.dropna() #删除空值 因为rolling(24) 会产生23行空值

#-------------------2.指定特征列与目标列 （哪个为x, 哪个为y）-------------------

for i in range(1,31): #构造目标列 lag1 1小时前pm2.5值 lag30 30小时前pm2.5值
    df[f"lag{i}"] = df["pm2.5"].shift(i) #生成lag特征

df = df.dropna() #删除空值

feature_cols = (
                   [ f"lag{i}" for i in range(1,31) ]
                    +
                   [
                     "TEMP",
                     "PRES",
                     "Iws",
                     "pm2.5_mean_24",
                     "pm2.5_max_24",
                     "pm2.5_std_24"
                   ]
                ) #特征集合为前30小时 温度 气压 风速 过去24小时平均污染水平 最严重污染水平 波动程度

X = df[feature_cols] #特征列
y = df["pm2.5"] #目标列

print(X.shape) #（677，36） 677个样本 36个特征
print(y.shape)

#---------------------3.划分数据集：训练集 测试集 （用于对比模型好坏）-----------------

"""
X_train, X_test, y_train, y_test = train_test_split(
                                                     X,
                                                     y,
                                                     test_size = 0.2,
                                                     random_state = 42
                                                    ) #这种划分方式模型评分92.9
"""

#时间序列会随机打乱 会把未来数据放进训练集 用train_test_split随机划分会造成信息泄露

split = int( len(X) * 0.8 ) #80%的数据用于得出模型， 20%的数据用于测试模型准确度 split=int(677*0.8)=int(541.6)=541

X_train = X[:split] #从0行取到541行
X_test = X[split:] #542行取到最后

y_train = y[:split] #从0行取到541行
y_test = y[split:] #542行取到最后

#进行标准化 在划分前标准化会信息提前泄露给模型 使得特征在同一个量纲上
from sklearn.preprocessing import StandardScaler #导入标准化函数
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) #fit_transform相当于先fit 在transform
X_test = scaler.transform(X_test) #fit_transform只能用于训练集 测试集必须transform
#fit会计算均值和标准差 在测试集上用fit 相当于让模型提前获得测试集的信息 属于数据泄露

print(X_train.shape) #共541个样本用于得出模型 36个特征
print(X_test.shape) #共136个样本用于测试 36个特征
print(y_train.shape)
print(y_test.shape)

#--------------------------------4.模型建立 拟合---------------------------

#建立随机森林模型

model = RandomForestRegressor(
                            n_estimators = 100, #100棵树
                            max_depth = 10, #最大深度为10
                            random_state = 42
                             )

model.fit(X_train, y_train) #训练随机森林回归

y_predict = model.predict(X_test) #预测
print("predict value", y_predict)
print("real value", y_test)

#--------------------------------5.模型评估 计算误差---------------------

mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print("均方误差:", mse) #mse=1390.82 rmse=mse*(1/2)=37.29>mae 大部分时候预测还行 存在少量误差较大的时刻
print("平均绝对误差:", mae) #mae=22.28 平均每次预测与真实值相差约22.28个pm2.5单位
print("R2=", r2) #R2=0.926 模型解释了约92.6%的数据波动


importance = model.feature_importances_ #获取特征重要性

feature_importance = pd.DataFrame({
                                   "feature": X.columns, #所有特征
                                   "importance":importance #对应的重要性
                                  })

feature_importance = feature_importance.sort_values(
                                                    by = "importance",
                                                    ascending=False
                                                   ) #给重要性排序，按照重要性降序排列
print(feature_importance)
#lag1的重要性达到95.8% 远高于其他特征，说明当前pm2.5序列具有极强的短期自相关特征。模型主要依赖前1小时浓度完成检测

#可视化特征重要性 前10个最重要的特征
top10 = feature_importance.head(10)
plt.figure( figsize=(10,6) )
plt.barh(
         top10["feature"],
         top10["importance"]
        )  #画水平柱状图
plt.title("Feature Importance")
plt.show()


#数据可视化

plt.figure( figsize=(12,6) ) #图形大小

plt.plot(y_test.values, label ="Real pm2.5") #真实值曲线
plt.plot(y_predict, label ="Predict pm2.5") #预测值曲线
plt.xlabel("Time index(hour)") #x轴名字
plt.ylabel("PM2.5 Concentration(ug/m3)") #y轴名字
plt.title("PM2.5 PredictionUsing Linear Regression") #图表名字

plt.legend()
plt.grid(True) #显示坐标网络
plt.show()