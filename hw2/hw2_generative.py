# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 01:49:14 2018

@author: Field Tien
"""
import numpy as np 
import pandas as pd 
import sys

def train_input(file):
    df=pd.read_csv(file,encoding='BIG5')
    workclass=pd.get_dummies(df['workclass'])
    workclass=workclass.drop(workclass.columns[0],axis=1)
    #need to drop one occu and edu
    education=pd.get_dummies(df['education'])
    education=education.drop(education.columns[0],axis=1)
    martial_status=pd.get_dummies(df['marital_status'])
    martial_status=martial_status.drop(martial_status.columns[0],axis=1)
    occupation=pd.get_dummies(df['occupation'])
    occupation.insert(0,'unknown_occu',workclass.iloc[:,0])
    occupation=occupation.drop(occupation.columns[1],axis=1)
    relationship=pd.get_dummies(df['relationship'])
    relationship=relationship.drop(relationship.columns[0],axis=1)
    race=pd.get_dummies(df['race'])
    race=race.drop(race.columns[0],axis=1)
    sex=pd.get_dummies(df['sex'])
    sex=sex.drop(sex.columns[0],axis=1)
    native=pd.get_dummies(df['native_country'])
    native=native.drop(native.columns[15],axis=1)
    native.insert(0,'unknown_native',native.iloc[:,0])
    native=native.drop(native.columns[1],axis=1)
    income=pd.get_dummies(df['income'])
    income=pd.get_dummies(df['income']).iloc[:,1]
    train=df[['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']]
    train=pd.concat([income,train,education,martial_status,occupation,relationship,race,sex],axis=1)
    for i in train.columns:
        train[i]=train[i].astype(np.float64)
    return(train)    
def test_input(file):
    tf=pd.read_csv(file,encoding='BIG5')   
    worktest=pd.get_dummies(tf['workclass'])
    worktest=worktest.drop(worktest.columns[0],axis=1)
    edutest=pd.get_dummies(tf['education'])
    edutest=edutest.drop(edutest.columns[0],axis=1)   
    martial_test=pd.get_dummies(tf['marital_status'])
    martial_test=martial_test.drop(martial_test.columns[0],axis=1)
    occupat_test=pd.get_dummies(tf['occupation'])
    occupat_test.insert(0,'unknown_occu',occupat_test.iloc[:,0])
    occupat_test=occupat_test.drop(occupat_test.columns[1],axis=1)
    relation_test=pd.get_dummies(tf['relationship'])
    relation_test=relation_test.drop(relation_test.columns[0],axis=1)
    race_test=pd.get_dummies(tf['race'])
    race_test=race_test.drop(race_test.columns[0],axis=1)
    sex_test=pd.get_dummies(tf['sex'])
    sex_test=sex_test.drop(sex_test.columns[0],axis=1)
    native_test=pd.get_dummies(tf['native_country'])
    native_test.insert(0,'unknown_native',native_test.iloc[:,0])
    native_test=native_test.drop(native_test.columns[1],axis=1)
    test=tf[['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']]
    test=pd.concat([test,edutest,martial_test,occupat_test,relation_test,race_test,sex_test],axis=1)
    for i in test.columns:
        test[i]=test[i].astype(np.float64)
    return(test)    
  
train=train_input(sys.argv[1])
x=np.matrix(train.drop(train.columns[0],axis=1))
y=np.matrix(train.iloc[:,0]).T  
test=test_input(sys.argv[2])
x_test=np.matrix(test)  
p_1=np.sum(y)/32561
p_0=1-p_1
t=train.sort_values(train.columns[0])
t=np.matrix(t)
x0=np.delete(t[range(24720),:],0,1)
x1=np.delete(t[range(24720,32561,1),:],0,1)
x0_mean=np.mean(x0, axis=0)
x1_mean=np.mean(x1, axis=0)
mean0=np.repeat(x0_mean,24720,axis=0)
mean1=np.repeat(x1_mean,7841,axis=0)
cov_0=np.dot((x0-mean0).T,(x0-mean0))/24720
cov_1=np.dot((x1-mean1).T,(x1-mean1))/7841
cov=p_0 * cov_0 + p_1 * cov_1

def constant(sigma):
    result=1/(np.sqrt(np.linalg.det(sigma))*(np.pi)**(37/2))
    return(result)
def pdf( x, mean, cov_inv,constant):
	result = constant * (np.exp( (-0.5) * np.dot( np.dot(x - mean, cov_inv), (x- mean).T)))[0,0]
	return (result)

constant=constant(cov)

predict=np.matrix(np.zeros((16281,1))) 
for i in range(16281):
    pc1=pdf(x_test[i,:], x1_mean,np.linalg.inv(cov),constant)
    pc0=pdf(x_test[i,:], x0_mean,np.linalg.inv(cov),constant)
    if pc0 < 10**-50:
    	pc0=10**-50
    if pc1 < 10**-50:
        pc1=10**-50	
    proportion=(pc0*p_0)/(pc1*p_1)
    pro=1/(1+proportion)
    if pro >0.5:
        predict[i,0]=1
predict=predict.astype(int)
y_hat=predict
test['label']=y_hat
y_hat=test.iloc[:,-1]
y_hat.index = list(range(1,16282))
y_hat.index = y_hat.index.set_names('id')
y_hat = y_hat.to_frame()
y_hat.to_csv(sys.argv[3])   




