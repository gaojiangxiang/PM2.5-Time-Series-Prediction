"""
实际序列预测项目：

数据为北京市2014年11月至12月每天每小时的pm2.5值 共730个数据

用2014年11月30个小时的pm2.5数值预测以后每小时的pm2.5值

均方误差: 18.848729018772186
平均绝对误差: 828.8169040247923
R2= 0.9550275684100062

"""

import sklearn
import pandas as pd #导入数据 处理
import random #导入随机数据
import matplotlib.pyplot as plt #导入图形，数据可视化
import numpy as np #导入数据，数组形式
from sklearn.linear_model import LinearRegression #导入线性回归包
from sklearn.model_selection import train_test_split #导入划分数据集的包
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score #导入模型评估公式 均方误差 平均绝对误差

#---------------------------1.获取数据集合， 处理数据-----------------------------------
df = pd.read_csv("air_quality.csv") #读取数据
df = df.iloc[1:731, 1:6] #只取年月日时 以及pm2.5的数据 即数据2-6列 以及2014年整个十一月的数据 即2-731行

print( df.to_string(index=False) ) # 去掉前面的索引
print(df.shape) #打印df的形状

print("pm2.5缺失值数量:", df["pm2.5"].isnull().sum()) #显示有14个缺失值 2014-11-20-21至2014-11-21-8
df["pm2.5"] = df["pm2.5"].interpolate() #在缺失值的位置进行插值，数据预处理方法
print("pm2.5缺失值数量:", df["pm2.5"].isnull().sum()) #显示有0个缺失值

#---------------------------2.指定特征列与目标列 （哪个为x, 哪个为y）-----------------------

for i in range(1,31): #构造目标列 lag1 1小时前pm2.5值 lag30 30小时前pm2.5值
    df[f"lag{i}"] = df["pm2.5"].shift(i) #生成lag特征

df = df.dropna() #删除空值

feature_cols = [f"lag{i}" for i in range(1,31)]
X = df[feature_cols]  #特征列 30小时的

y = df["pm2.5"] #目标列

print(X.shape) #（700，30） 700个样本 30个特征
print(y.shape)

#-----------------------------3.划分数据集：训练集 测试集 （用于对比模型好坏）----------------------

"""
X_train, X_test, y_train, y_test = train_test_split(
                                                     X,
                                                     y,
                                                     test_size = 0.2,
                                                     random_state = 42
                                                    ) #这种划分方式模型评分92.9
"""

#时间序列会随机打乱 会把未来数据放进训练集 用train_test_split随机划分会造成信息泄露 这种分划分方式模型评分95.5 这种更好

split = int( len(X) * 0.8 ) #80%的数据用于得出模型， 20%的数据用于测试模型准确度 split=int(700*0.8)=int(560)=560

X_train = X[:split] #从0行取到559行
X_test = X[split:] #560行取到最后

y_train = y[:split] #从0行取到559行
y_test = y[split:] #560行取到最后

print(X_train.shape) #共560个样本用于得出模型 30个特征
print(X_test.shape) #共140个样本用于测试 30个特征
print(y_train.shape)
print(y_test.shape)

#---------------------------------4.模型建立 拟合-----------------------------------

#导入线性回归包
model = LinearRegression()
model.fit(X_train, y_train) #训练线性回归

weight_df = pd.DataFrame({
                          "Lag_Hour":range(30,0,-1),
                          "Weight":model.coef_
                         })
print( weight_df.to_string(index=False) ) # 去掉前面的索引 #打印每小时的权重 斜率

print("截距:",model.intercept_) #截距  #y = w(斜率)x + b(截距)
#y = a1x1+a2x2+...+a30x30 + b

y_predict = model.predict(X_test) #预测
print("predict value", y_predict)
print("real value", y_test)

#----------------------------------5.模型评估 计算误差------------------------

mae = mean_squared_error(y_test, y_predict)
mse = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print("均方误差:", mse) # mse=861.59 rmse=mse*(1/2)=29.35>mae 大部分时候预测还行 存在少量误差较大的时刻
print("平均绝对误差:", mae) #mae=18.69 平均每次预测与真实值相差约18.7个pm2.5单位
print("R2=", r2) #R2=0.956 模型解释了约95.6%的数据波动

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





