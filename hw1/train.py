# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 02:01:04 2018

@author: Field Tien
"""

import numpy as np 
import pandas as pd 
def remove_outlier(df_in, col_name):
    q3 = df_in[col_name].quantile(0.9975)
    df_out = df_in.loc[ df_in[col_name] < q3]
    return (df_out)   
def traing_data_input(file):
    train=pd.read_csv(file,encoding='cp950')
    cross=train['測項'].tolist()
    threedex=[[x for i in range(1,13) for x in np.repeat(i,360).tolist()],
               [x for i in range(1,21) for x in np.repeat(i,18).tolist()]*12,
               cross]
    tuples=list(zip(*threedex))
    index = pd.MultiIndex.from_tuples(tuples, names=['month', 'date','kind'])
    train=train.set_index(index)
    train=train.drop(['測站','測項','日期'],axis=1)
    train.columns=train.columns.astype(int)
    train=train.unstack(level=[-2,-3])
    train.ix['RAINFALL']=train.ix['RAINFALL'].apply(lambda x: 0 if x=='NR' else x)
    for i in train.columns:
        train[i]= train[i].astype(np.float64)
    train=train.unstack()
    train=train.unstack(2)
    train=train.unstack(1)
    train=train.unstack(0)
    for i in np.arange(0,5760,1):
        if np.sum(train.iloc[:,i])==0:
            train.iloc[:,i]=train.iloc[:,i-1]
    for i in np.arange(0,5760,1):
        if np.matrix(train.loc['AMB_TEMP'][i])[0,0] == 0:
            train.loc['AMB_TEMP'][i]=train.ix['AMB_TEMP'][i-1]
        if np.matrix(train.loc['PM2.5'][i])[0,0] <= 0:
            train.loc['PM2.5'][i]=train.ix['PM2.5'][i-1]
        if np.matrix(train.loc['PM10'][i])[0,0] <= 0:
            train.loc['PM10'][i]=train.ix['PM10'][i-1]        
        if np.matrix(train.loc['CO'][i])[0,0] <= 0:
            train.loc['CO'][i]=train.ix['CO'][i-1]               
        if np.matrix(train.loc['CH4'][i])[0,0] <= 0:
            train.loc['CH4'][i]=train.ix['CH4'][i-1]      
        if np.matrix(train.loc['NOx'][i])[0,0] <= 0:
            train.loc['NOx'][i]=train.ix['NOx'][i-1]  
        if np.matrix(train.loc['O3'][i])[0,0] <= 0:
            train.loc['O3'][i]=train.ix['O3'][i-1] 
        if np.matrix(train.loc['RH'][i])[0,0] <= 0:
            train.loc['RH'][i]=train.ix['RH'][i-1]
        if np.matrix(train.loc['NO2'][i])[0,0] <= 0:
            train.loc['NO2'][i]=train.ix['NO2'][i-1]      
   
    train.ix['sinWIND']=np.sin(np.pi/360*train.ix['WIND_DIREC'])
    train.ix['cosWIND']=np.cos(np.pi/360*train.ix['WIND_DIREC'])
    train.ix['sinWINDspeed']=np.multiply(np.sin(np.pi/360*train.ix['WIND_DIREC']),train.ix['WIND_SPEED'])
    train.ix['cosWINDspees']=np.multiply(np.cos(np.pi/360*train.ix['WIND_DIREC']),train.ix['WIND_SPEED'])
    train.ix['sinWINDHR']=np.sin(np.pi/360*train.ix['WD_HR'])
    train.ix['cosWINDHR']=np.cos(np.pi/360*train.ix['WD_HR'])
    
    train=train.drop(['WIND_DIREC','WD_HR'],axis=0)
    train=train.stack(0)
    lag9=train.drop([(20,23),(20,22),(20,21),(20,20),(20,19),(20,18),(20,17),(20,16),(20,15)],axis=1).stack([0,1]).unstack(0)
    lag8=train.drop([(20,23),(20,22),(20,21),(20,20),(20,19),(20,18),(20,17),(20,16),(1,0)],axis=1).stack([0,1]).unstack(0)
    lag7=train.drop([(20,23),(20,22),(20,21),(20,20),(20,19),(20,18),(20,17),(1,1),(1,0)],axis=1).stack([0,1]).unstack(0)
    lag6=train.drop([(20,23),(20,22),(20,21),(20,20),(20,19),(20,18),(1,2),(1,1),(1,0)],axis=1).stack([0,1]).unstack(0)
    lag5=train.drop([(20,23),(20,22),(20,21),(20,20),(20,19),(1,3),(1,2),(1,1),(1,0)],axis=1).stack([0,1]).unstack(0)
    lag4=train.drop([(20,23),(20,22),(20,21),(20,20),(1,4),(1,3),(1,2),(1,1),(1,0)],axis=1).stack([0,1]).unstack(0)
    lag3=train.drop([(20,23),(20,22),(20,21),(1,5),(1,4),(1,3),(1,2),(1,1),(1,0)],axis=1).stack([0,1]).unstack(0)
    lag2=train.drop([(20,23),(20,22),(1,6),(1,5),(1,4),(1,3),(1,2),(1,1),(1,0)],axis=1).stack([0,1]).unstack(0)
    lag1=train.drop([(20,23),(1,7),(1,6),(1,5),(1,4),(1,3),(1,2),(1,1),(1,0)],axis=1).stack([0,1]).unstack(0)
    y=train.drop([(1,8),(1,7),(1,6),(1,5),(1,4),(1,3),(1,2),(1,1),(1,0)],axis=1).stack([0,1]).unstack(0)['PM2.5']
    lag1.index=list(np.arange(0,5652,1))
    lag2.index=list(np.arange(0,5652,1))
    lag3.index=list(np.arange(0,5652,1))
    lag4.index=list(np.arange(0,5652,1))
    lag5.index=list(np.arange(0,5652,1))
    lag6.index=list(np.arange(0,5652,1))
    lag7.index=list(np.arange(0,5652,1))
    lag8.index=list(np.arange(0,5652,1))
    lag9.index=list(np.arange(0,5652,1))
    y.index=list(np.arange(0,5652,1))
    a=list(lag9.columns)
    lag1.columns=['lag1_' +str(i) for i in a]
    lag2.columns=['lag2_' +str(i) for i in a]
    lag3.columns=['lag3_' +str(i) for i in a]
    lag4.columns=['lag4_' +str(i) for i in a]
    lag5.columns=['lag5_' +str(i) for i in a]
    lag6.columns=['lag6_' +str(i) for i in a]
    lag7.columns=['lag7_' +str(i) for i in a]
    lag8.columns=['lag8_' +str(i) for i in a]
    lag9.columns=['lag9_' +str(i) for i in a]
    result=pd.concat([y,lag9,lag8,lag7,lag6,lag5,lag4,lag3,lag2,lag1],axis=1)
    for i in ['PM2.5']+['lag'+str(i)+'_PM2.5' for i in range(1,10,1)]:
        result=remove_outlier(result, i)
    return(result)
train=traing_data_input('C:/Users/Field Tien/Desktop/hw1/train.csv') 
x=np.matrix(train.drop(['PM2.5'],axis=1))
y=np.matrix(train['PM2.5']).T  
def lingrad(y,x,learn,itera):
    x=np.concatenate((np.matrix(np.repeat(1,np.array(x.shape)[0])).T, x), axis=1)
    n=np.array(x.shape)[0]
    p=np.array(x.shape)[1]
    beta=np.matrix(np.repeat(0,np.array(x.shape)[1])).T
    ada=0
    for i in np.arange(itera):
        loss=y-np.dot(x,beta)
        grad= -2*np.dot(x.T,loss)
        ada1= ((ada**2)+np.asscalar(np.dot(grad.T,grad)))**(1/2)
        ada0=ada1
        beta2=beta-(learn/ada1)*grad
        if np.linalg.norm(grad)<10**-1:
            break
        else:
            beta=beta2
    return(beta)  
def linfit(x,beta):
    x=np.concatenate((np.matrix(np.repeat(1,np.array(x.shape)[0])).T, x), axis=1)
    n=np.array(x.shape)[0]
    p=np.array(x.shape)[1]
    y_hat=np.dot(x,beta)
    return(y_hat)
def linloss(y,x,beta):
    x=np.concatenate((np.matrix(np.repeat(1,np.array(x.shape)[0])).T, x), axis=1)
    n=np.array(x.shape)[0]
    p=np.array(x.shape)[1]
    y_hat=np.dot(x,beta)
    loss=y- y_hat
    meanloss=np.asscalar(np.dot(loss.T,loss))
    return(meanloss)    
def cross_validation(y,x,kfold,graditera,learn):
    n=np.array(x.shape)[0]
    split=np.append(np.array(np.repeat(int(n/kfold),kfold-1)),(n-(kfold-1)*int(n/kfold)))
    cumsplit=np.cumsum(split)
    lossmean=np.repeat(0,kfold)
    for i in np.arange(0,kfold,1):
        x_train=np.delete(x,np.arange(0+i*split[0],cumsplit[i]),axis=0)
        y_train=np.delete(y,np.arange(0+i*split[0],cumsplit[i]),axis=0)
        x_test=x[np.arange(0+i*split[0],cumsplit[i]),:]
        y_test=y[np.arange(0+i*split[0],cumsplit[i]),:]
        beta=lingrad(y_train,x_train,learn,graditera)
        lossmean[i]=linloss(y_test,x_test,beta)
    rmse= ((np.sum(lossmean))/n)**(1/2) 
    return(rmse)      
beta=lingrad(y,x,0.01,5000)
def test_data_input(file):
    test=pd.read_csv(file,encoding='cp950',header=None)
    test=test.set_index([0,1])
    test=test.unstack(0)
    test.ix['RAINFALL']=test.ix['RAINFALL'].apply(lambda x: 0 if x=='NR' else x)
    for i in test.columns:
        test[i]= test[i].astype(np.float64)
    test.ix['sinWIND']=np.sin(np.pi/360*test.ix['WIND_DIREC']) 
    test.ix['cosWIND']=np.cos(np.pi/360*test.ix['WIND_DIREC'])  
    test.ix['sinWINDspeed']=np.multiply(np.sin(np.pi/360*test.ix['WIND_DIREC']),test.ix['WIND_SPEED'])
    test.ix['cosWINDspees']=np.multiply(np.cos(np.pi/360*test.ix['WIND_DIREC']),test.ix['WIND_SPEED'])
    test.ix['sinWINDHR']=np.sin(np.pi/360*test.ix['WD_HR'])
    test.ix['cosWINDHR']=np.cos(np.pi/360*test.ix['WD_HR'])
    test=test.drop(['WIND_DIREC','WD_HR'],axis=0)
    test=test.stack(1)
    test.columns=list(np.arange(0,9,1))
    test=test.stack()
    test=test.unstack(1)
    test['AMB_TEMP'][[535,536,537,538]]=np.repeat(17,4)
    test['CH4'][[523,526]]=[1.7,1.8]
    test['NO2'][[1701,1702,1863,1864,1865,1866]]=[15,15,14,14,14,14]
    for i in range(2340):
        if  test['NO2'][i] <=0:
            test['NO2'][i]=test['NO2'][i-1]
    test['NOx'][[1701,1703]]=[16,16]
    for i in range(2340):
        if  test['NOx'][i] <=0:
            test['NOx'][i]=test['NOx'][i-1]
    test['O3'][1451,1452]=[52,52]
    test['PM10'][[378,2161]]=[112,82]      
    for i in range(2340):
        if  test['PM10'][i] <=0:
            test['PM10'][i]=test['PM10'][i-1]
    test['PM2.5'][2160]=38  
    for i in range(2340):
        if  test['PM2.5'][i] <=0:
            test['PM2.5'][i]=test['PM2.5'][i-1]
    for i in range(2340):
        if  test['RH'][i] <=0:
            test['RH'][i]=test['RH'][i-1]    
    test=test.stack()
    test=test.unstack(0)
    test=test.T
    return(test)        
test=test_data_input('C:/Users/Field Tien/Desktop/hw1/test.csv')    
x_test=np.matrix(test)     
x_test=np.concatenate((np.matrix(np.repeat(1,np.array(x_test.shape)[0])).T, x_test), axis=1)
y_hat=np.dot(x_test,beta)            
test['reg']=y_hat
y_hat=test.iloc[:,-1]
y_hat.index = y_hat.index.set_names('id')
y_hat = y_hat.to_frame()
y_hat.columns = y_hat.columns.droplevel(level=-1)
y_hat = y_hat.rename(columns = {'reg':'value'})
y_hat.to_csv('reg.csv')            
            