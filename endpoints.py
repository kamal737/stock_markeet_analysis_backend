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
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import datetime
import jsonify
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
from twilio.rest import Client 
plt.style.use('fivethirtyeight')
import yfinance as yf
yf.pdr_override()





def project_api_routes(endpoints):
 

 @endpoints.route('/otp',methods=['POST'])

 def get_otp():
  
  if request.method == 'POST':
    global otpnum
    otpnum=request.values['otp']
    print(otpnum)
    account_sid = 'ACb21501b89a7bd89c0ccfbfd6e26ff131' 
    auth_token = 'd6479d113a45a295f4d9abb104c67bff' 
    client = Client(account_sid, auth_token) 
    otp2='Please login using this otp {otpnum}'
    message = client.messages.create(  
                                    from_='+15677024271',
                                    body = otp2,
                                    to='+919361385989' 
                                ) 
      
    print(message.sid)
    

    return 'otp sent successfully'
 

 @endpoints.route('/upload',methods=['POST','GET'])

 def upload_file():

  if request.method=="POST":
    global stock
    global sdate
    global edate
    stock=request.values['stock']
    sdate=request.values['sdate']
    edate=request.values['edate']
    print(stock)
    print(sdate)
    print(edate)
    # start = sdate
    # end = edate
    df = yf.download('GOOGL', start=sdate, end=edate)
    prediction_range = .6
    plt.figure(figsize=(16,8))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price USD($)',fontsize=18)
    data=df.filter(['Close'])
    dataset=data.values
    training_data_len= math.ceil(len(dataset) * prediction_range)

    #scale the data
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)
    scaled_data
    training_data_len, len(dataset)
    train_data=scaled_data[0:training_data_len,:]
    x_train =[]
    y_train=[]
    for i in range(60,len(train_data)):
      x_train.append(train_data[i-60:i,0])
      y_train.append(train_data[i,0])
      if i<=60:

        print(x_train)
        print(y_train)
        print()
    x_train= np.array(x_train)
    y_train=np.array(y_train)
    x_train.shape   
    x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_train.shape
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(50,return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(x_train,y_train,batch_size=1,epochs=2)
    test_data=scaled_data[training_data_len - 60:,:]
    x_test=[]
    y_test=dataset[training_data_len:,:]
    for i in range(60,len(test_data)):
      x_test.append(test_data[i-60:i,0])
    x_test=np.array(x_test)  
    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    predictions=model.predict(x_test)
    predictions=scaler.inverse_transform(predictions)
    rmse=np.sqrt( np.mean((predictions - y_test)**2))
    rmse
    train=data[:training_data_len]
    valid=data[training_data_len:]
    valid['Predictions']=predictions

    #visualize the data
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price USD($)',fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train','Val','Predictions'],loc='lower right')
    plt.show()

    return 'kamalakannan'
 return endpoints