from unicodedata import name
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import logging
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request
from tkinter import E
from flask_pymongo import pymongo
import warnings





def project_api_routes(endpoints):

 @endpoints.route('/login',methods=['POST','GET'])
 def login():
  global f
  if request.method=='POST':
   f=request.values['name']
   return 'data posted succesfully'
  if request.method=='GET':
   return f
  
 @endpoints.route('/upload',methods=['POST','GET'])
 def upload_file():
  if request.method=="POST":
    global df
    global a
    f=request.files['file']
    a=request.values['number']
    f.save(secure_filename(f.filename))
    df=pd.read_csv(f.filename)
    return 'file uploaded successfully'
  if request.method=="GET":
   a=int(a)
   df.head()
   df.columns=["Month","Sales"]
   df.head()
   df.isnull().sum()
   df.tail()
   df['Month']=pd.to_datetime(df['Month'])
   df.set_index('Month',inplace=True)
   df.describe()
   df.info()
   df.shape
   from statsmodels.tsa.stattools import adfuller
   test_result=adfuller(df['Sales'])
   def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
   adfuller_test(df['Sales'])
   df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)
   df['Sales'].shift(1)
   df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)
   adfuller_test(df['Seasonal First Difference'].dropna())
   df['Seasonal First Difference'].plot()
   from pandas.plotting import autocorrelation_plot
   autocorrelation_plot(df['Sales'])
   from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
   fig = plt.figure(figsize=(12,8))
   ax1 = fig.add_subplot(211)
   fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
   ax2 = fig.add_subplot(212)
   fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)
   from statsmodels.tsa.arima.model import ARIMA
   model=ARIMA(df['Sales'],order=(1,1,1))
   model_fit=model.fit()
   warnings.filterwarnings("ignore")
   model_fit.summary()
   df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
   df[['Sales','forecast']].plot(figsize=(12,8))
   model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
   results=model.fit()
   df['forecast']=results.predict(start=90,end=103,dynamic=True)
   df[['Sales','forecast']].plot(figsize=(12,8))
   from pandas.tseries.offsets import DateOffset
   future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,a)]
   future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)
   future_datest_df.tail()
   future_df=pd.concat([df,future_datest_df])
   future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)  
   dff = future_df.loc[future_df['forecast'].notnull()]
   return dff['forecast'].to_json(orient="records")
 return endpoints


     


   


 