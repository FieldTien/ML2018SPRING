# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 01:13:02 2018

@author: Field Tien
"""

import numpy as np 
import pandas as pd 
import sys
import pickle
def test_data_input(file):
    test=pd.read_csv(file,encoding='cp950',header=None)
    test=test.set_index([0,1])
    test=test.unstack(0)
    test.loc['RAINFALL']=test.loc['RAINFALL'].apply(lambda x: 0 if x=='NR' else x)
    for i in test.columns:
        test[i]= test[i].astype(np.float64)
    
    test.loc['sinWIND']=np.sin(np.pi/360*test.loc['WIND_DIREC']) 
    test.loc['cosWIND']=np.cos(np.pi/360*test.loc['WIND_DIREC'])  
    test.loc['sinWINDspeed']=np.multiply(np.sin(np.pi/360*test.loc['WIND_DIREC']),test.loc['WIND_SPEED'])
    test.loc['cosWINDspees']=np.multiply(np.cos(np.pi/360*test.loc['WIND_DIREC']),test.loc['WIND_SPEED'])
    test.loc['sinWINDHR']=np.sin(np.pi/360*test.loc['WD_HR'])
    test.loc['cosWINDHR']=np.cos(np.pi/360*test.loc['WD_HR'])
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
test=test_data_input(sys.argv[1])
x_test=np.matrix(test)    
x_test=x_test[:,range(22,198,1)]
x_test=np.concatenate((np.matrix(np.repeat(1,np.array(x_test.shape)[0])).T, x_test), axis=1)
file=open('best.pickle','rb')
coe=pickle.load(file)
file.close()
beta75=np.matrix(coe['q75'])   
beta99=np.matrix(coe['q99'])   
beta=np.matrix(coe['final']) 
did=np.dot(x_test,beta99)-np.dot(x_test,beta75)
label=np.matrix(np.zeros((260,1)))
for i in range(260):
    if did[i] > 10 :
        label[i,0]=1    
x_test=np.concatenate((x_test,label),axis=1)
y_hat=np.dot(x_test,beta)        
test['reg']=y_hat
y_hat=test.iloc[:,-1]
y_hat.index = y_hat.index.set_names('id')
y_hat = y_hat.to_frame()
y_hat.columns = y_hat.columns.droplevel(level=-1)
y_hat = y_hat.rename(columns = {'reg':'value'})
y_hat.to_csv(sys.argv[2],header=True)    
    