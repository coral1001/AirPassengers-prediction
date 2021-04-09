# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import dateparser
from datetime import datetime
import matplotlib.pylab as plt
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv('d:/airp/AirPassengers.csv')
print(data.head())

dateparse = lambda x:pd.datetime.strptime(x,'%Y/%m/%d')
#---其中parse_dates 表明选择数据中的哪个column作为date-time信息，
#---index_col 告诉pandas以哪个column作为 index
#--- date_parser 使用一个function(本文用lambda表达式代替)，使一个string转换为一个datetime变量
'''
#取对数：可以消除时间序列中的异方差；而取差分：能消除时间序列的非平稳性（尽可能多次的差分，必定能得到平稳时间序列

'''
data = pd.read_csv('d:/airp/AirPassengers.csv',parse_dates=['time'],date_parser=dateparse)
data = data.set_index('time')
#plt.plot(data.value)

ts_log = np.log(data['value']) #除时间序列中的异方差

decomposition = seasonal_decompose(ts_log,freq=12)
trend = decomposition.trend #趋势
seasonal = decomposition.seasonal  #季节性
residual = decomposition.resid     #残差序列
residual.dropna(inplace=True)

dftest = ts.adfuller(residual)
dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Obserfvisions Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value%s'%key] = value
print(dfoutput)

from statsmodels.tsa.arima_model import ARIMA
model_ARIMA = ARIMA(residual,(2,0,2)).fit(disp=-1, method='css')
predictions_ARIMA = model_ARIMA.predict(start='1950-01',end='1962-04')
#plt.plot(residual)
#plt.plot(predictions_ARIMA)
#print(trend)
#print(seasonal)
#plt.plot(seasonal)

#trend拟合，应为trend是线性的，因此使用线性回归模型拟合去预测1960之后的数据
trend.dropna(inplace=True)
from sklearn.linear_model import LinearRegression
#模拟数据
x = pd.Series(range(trend.size),index=trend.index)
x = x.to_frame()
#创建模型
linreg = LinearRegression()
#拟合模型
linereg = linreg.fit(x, trend)
x = pd.Series(range(0,154),index=(pd.period_range('1949-07',periods=154,freq = 'M')))
x = x.to_frame()
#预测
res_predict = linereg.predict(x)
trend2 = pd.Series(res_predict,index=x.index).to_timestamp() 
#plt.plot(trend,color='blue')
#plt.plot(trend2,color='red')
'''
#扩展seasonal
index1 = pd.period_range('1949-01',periods=160,freq = 'M')
index1= index1.to_datetime()
seasonal = seasonal.reindex(index1)
seasonal = seasonal.shift(24)
plt.plot(seasonal)
'''
model_ARIMA = ARIMA(residual,(2,0,2)).fit(disp=-1, method='css')
predictions_ARIMA = model_ARIMA.predict(start='1950-01',end='1962-04')
predictions_ARIMA = predictions_ARIMA.add(trend2,fill_value=0).add(seasonal,fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA)
plt.plot(data['value'],color='blue')
plt.plot(predictions_ARIMA,color='red')





