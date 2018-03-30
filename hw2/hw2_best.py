# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 22:36:33 2018

@author: Field Tien
"""
from sklearn import svm
import numpy as np 
import pandas as pd 
import pickle
import sys
def test_input(file):
    tf=pd.read_csv(file,encoding='BIG5')   
    worktest=pd.get_dummies(tf['workclass'])
    edutest=pd.get_dummies(tf['education'])
    martial_test=pd.get_dummies(tf['marital_status'])
    occupat_test=pd.get_dummies(tf['occupation'])
    relation_test=pd.get_dummies(tf['relationship'])
    race_test=pd.get_dummies(tf['race'])
    sex_test=pd.get_dummies(tf['sex'])
    native_test=pd.get_dummies(tf['native_country'])
    native_test.insert(0,'unknown_native',native_test.iloc[:,0])
    native_test=native_test.drop(native_test.columns[1],axis=1)
    tf['power_age']=tf['age'].pow(2)
    test=tf[['age','power_age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']]
    test=pd.concat([test,worktest,edutest,martial_test,occupat_test,relation_test,race_test,sex_test,native_test],axis=1)
    for i in test.columns:
        test[i]=test[i].astype(np.float64)
    return(test)    
test=test_input(sys.argv[1])
test['fnlwgt']=test['fnlwgt'].apply(lambda x: (x-12285)/(1484705-12285))
test['capital_gain']=test['capital_gain'].apply(lambda x: (x)/99999)
test['capital_loss']=test['capital_loss'].apply(lambda x: (x)/4356)
test['age']=test['age'].apply(lambda x: (x-17)/73)
test['power_age']=test['power_age'].apply(lambda x: (x-289)/5329)   
x_test=np.matrix(test)    

file=open('SVM.pickle','rb')
cluster=pickle.load(file)
file.close()  
SVM0=cluster['SVM0']
SVM1=cluster['SVM1']
SVM2=cluster['SVM2']
SVM3=cluster['SVM3']
SVM4=cluster['SVM4']
SVM5=cluster['SVM5']
SVM6=cluster['SVM6']
SVM7=cluster['SVM7']
SVM8=cluster['SVM8']
SVM9=cluster['SVM9'] 
SVM10=cluster['SVM10'] 
y_hat0=np.matrix(SVM0.predict(x_test).astype(int)).T
y_hat1=np.matrix(SVM1.predict(x_test).astype(int)).T
y_hat2=np.matrix(SVM2.predict(x_test).astype(int)).T
y_hat3=np.matrix(SVM3.predict(x_test).astype(int)).T
y_hat4=np.matrix(SVM4.predict(x_test).astype(int)).T
y_hat5=np.matrix(SVM5.predict(x_test).astype(int)).T
y_hat6=np.matrix(SVM6.predict(x_test).astype(int)).T
y_hat7=np.matrix(SVM7.predict(x_test).astype(int)).T
y_hat8=np.matrix(SVM8.predict(x_test).astype(int)).T
y_hat9=np.matrix(SVM9.predict(x_test).astype(int)).T
y_hat10=np.matrix(SVM10.predict(x_test).astype(int)).T
Y=y_hat0+y_hat1+y_hat2+y_hat3+y_hat4+y_hat5+y_hat6+y_hat7+y_hat8+y_hat9+y_hat10
y_hat=np.matrix(np.zeros((16281,1)))
for i in range(16281):
    if Y[i] >= 6:
        y_hat[i,]=1
y_hat=y_hat.astype(int)
test['label']=y_hat
y_hat=test.iloc[:,-1]
y_hat.index = list(range(1,16282))
y_hat.index = y_hat.index.set_names('id')
y_hat = y_hat.to_frame()
y_hat.to_csv(sys.argv[2])           

