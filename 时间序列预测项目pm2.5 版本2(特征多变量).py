"""
时间序列预测项目：

数据为北京市2014年11月至12月每天每小时的pm2.5值 共730个数据

用2014年11月30个小时的pm2.5数值 温度 气压 风速 过去24小时平均污染水平 最严重污染水平 波动程度 预测以后每小时的pm2.5值

多变量线性回归预测 外加特征工程（增加气象特征）

pm2.5不仅受历史浓度影响 还受到气象条件的影响

均方误差: 796.2292892253363
平均绝对误差: 18.40669571579595
R2= 0.957985197975975 （相较v1提升了0.002）

"""

import sklearn
import pandas as pd #导入数据 处理
import random #导入随机数据
import matplotlib.pyplot as plt #导入图形，数据可视化
import numpy as np #导入数据，数组形式
from sklearn.linear_model import LinearRegression #导入线性回归包
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

#时间序列会随机打乱 会把未来数据放进训练集 用train_test_split随机划分会造成信息泄露 这种分划分方式模型评分95.5 这种更好

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

#导入线性回归包
model = LinearRegression()
model.fit(X_train, y_train) #训练线性回归

#重要性分析 查看哪个特征对pm2.5的预测影响最大
importance = pd.DataFrame({
                          "Feature": feature_cols, #所有的特征
                          "Coefficient":model.coef_ # 每个特征对应的系数 即斜率（权重）
                         })
importance["Abs"] = importance["Coefficient"].abs() #取每个特征系数的绝对值

importance = importance.sort_values(
    by = "Abs", #按照哪一列排序
    ascending = False #是否升序 从大到小
) #按照重要性大小排序
print( importance.head(10) ) # 取前十个重要的特征 影响最大的

y_predict = model.predict(X_test) #预测
print("predict value", y_predict)
print("real value", y_test)

#--------------------------------5.模型评估 计算误差---------------------

mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print("均方误差:", mse) #mse=796.22 rmse=mse*(1/2)=28.21>mae 大部分时候预测还行 存在少量误差较大的时刻
print("平均绝对误差:", mae) #mae=18.4 平均每次预测与真实值相差约18.4个pm2.5单位
print("R2=", r2) #R2=0.957 模型解释了约95.7%的数据波动

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
