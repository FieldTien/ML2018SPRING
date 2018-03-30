# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 16:41:46 2018

@author: Field Tien
"""
import numpy as np 
import pandas as pd 
import sys
def logit(y,x,learn,itera):
    n=np.array(x.shape)[0]
    p=np.array(x.shape)[1]
    beta=np.matrix(np.repeat(0,np.array(x.shape)[1])).T
    ada=np.matrix(np.repeat(0,np.array(x.shape)[1])).T
    for i in np.arange(itera):
        z=np.dot(x,beta)
        sigmoid=1/(1+np.exp(-z))
        loss=sigmoid-y
        grad= np.dot(x.T,loss)
        ada1= np.sqrt(np.square(ada)+np.square(grad))
        ada1=np.apply_along_axis(lambda x: 10**(-6) if x==0 else x,1,ada1).T
        beta=beta-np.multiply((learn/ada1),grad)
        ada=ada1
        
    return(beta)    
def predict(x,beta):
    #x=np.concatenate((np.matrix(np.repeat(1,np.array(x.shape)[0])).T, x), axis=1)
    z=np.dot(x,beta)
    sigmoid=1/(1+np.exp(-z))
    pre=np.apply_along_axis(lambda x: 1 if x>=0.5 else 0,1,sigmoid)
    pre=np.matrix(pre).T
    return(pre)
    
    
def res(y,x,learn,itera):
    x=np.concatenate((np.matrix(np.repeat(1,np.array(x.shape)[0])).T, x), axis=1)
    n=np.array(x.shape)[0]
    p=np.array(x.shape)[1]
    beta=np.matrix(np.repeat(0,np.array(x.shape)[1])).T
    ada=np.matrix(np.repeat(0,np.array(x.shape)[1])).T
    res=np.matrix(np.zeros((itera,1)))
    for i in np.arange(itera):
        z=np.dot(x,beta)
        sigmoid=1/(1+np.exp(-z))
        loss=sigmoid-y
        grad= np.dot(x.T,loss)/n
        ada1= np.sqrt(np.square(ada)+np.square(grad))
        ada1=np.apply_along_axis(lambda x: 10**(-6) if x==0 else x,1,ada1).T
        beta1=beta-np.multiply((learn/ada1),grad)
        res[i,0]=np.linalg.norm(grad)
        beta=beta1
        ada=ada1
    return(res)  
    
def mini(y,x,learn,itera):
    x=np.concatenate((np.matrix(np.repeat(1,np.array(x.shape)[0])).T, x), axis=1)
    n=np.array(x.shape)[0]
    p=np.array(x.shape)[1]
    beta=np.matrix(np.repeat(0,np.array(x.shape)[1])).T
    ada=np.matrix(np.repeat(0,np.array(x.shape)[1])).T
    res=np.matrix(np.zeros((itera,1)))
    index=np.arange(0,32561,1)
    for i in np.arange(itera):
        minibat=np.random.permutation(index)[range(5000)]
        x_m=x[minibat]
        y_m=y[minibat]
        z=np.dot(x_m,beta)
        sigmoid=1/(1+np.exp(-z))
        loss=sigmoid-y_m
        grad= np.dot(x_m.T,loss)/len(y_m)
        ada1= np.sqrt(np.square(ada)+np.square(grad))
        ada1=np.apply_along_axis(lambda x: 10**(-6) if x==0 else x,1,ada1).T
        beta1=beta-np.multiply((learn/ada1),grad)
        res[i,0]=np.linalg.norm(grad)
        beta=beta1
        ada=ada1
    return(res)        

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
        beta=logit(y_train,x_train,learn,graditera)
        pre=predict(x_test,beta)
        right=0
        for j in range(len(y_test)):
            if pre[i,0]==y_test[i,0]:
               right=right+1 
        lossmean[i]=right
    rmse= np.sum(lossmean)/n
    return(rmse) 
def train_input(file):
    df=pd.read_csv(file,encoding='BIG5')
    workclass=pd.get_dummies(df['workclass'])
    #need to drop one occu and edu
    education=pd.get_dummies(df['education'])
    education['ungraduate']=education[' 10th']+education[' 11th']+education[' 12th']+education[' HS-grad']
    education['junior']=education[' 1st-4th']+education[' 5th-6th']+education[' 7th-8th']+education[' 9th']
    education=education.drop([' 10th',' 11th',' 12th',' HS-grad',' 1st-4th',' 5th-6th',' 7th-8th',' 9th'],axis=1)
    martial_status=pd.get_dummies(df['marital_status'])
    occupation=pd.get_dummies(df['occupation'])
    relationship=pd.get_dummies(df['relationship'])
    race=pd.get_dummies(df['race'])
    sex=pd.get_dummies(df['sex'])
    native=pd.get_dummies(df['native_country'])
    native=native.drop(native.columns[15],axis=1)
    native.insert(0,'unknown_native',native.iloc[:,0])
    native=native.drop(native.columns[1],axis=1)
    native['poor']=native[' Cambodia']+native[' China']+native[' Columbia']+native[' Cuba']+native[' Dominican-Republic']+native[' Ecuador']+native[' El-Salvador']+native[' Guatemala']+native[' Haiti']+native[' Honduras']+native[' India']+native[' Iran']+native[' Jamaica']+native[' Laos']+native[' Mexico']+native[' Peru']+native[' Philippines']+native[' Thailand']+native[' Vietnam']+native[' Yugoslavia']+native[' Nicaragua']
    native['middle']=native[' Greece']+native[' Portugal']+native[' Trinadad&Tobago']+native[' Poland']+native[' South']+native[' Hungary']+native[' Taiwan']+native[' Outlying-US(Guam-USVI-etc)']+native[' Scotland']+native[' Puerto-Rico']+native[' England']+native[' France']+native[' Italy']+native[' Japan']+native[' Ireland']+native[' Germany']+native[' Hong']+native[' Canada']
    native['USA']=native[' United-States']
    native=pd.concat([native['USA'],native['middle'],native['poor']],axis=1)
    
    income=pd.get_dummies(df['income'])
    income=pd.get_dummies(df['income']).iloc[:,1]
    df['power_age']=df['age'].pow(2)
    df['10hour']=df['hours_per_week'].apply(lambda x: 1 if x <10 else 0 )
    df['2030hour']=df['hours_per_week'].apply(lambda x: 1 if x <30 and x>=10 else 0 )
    df['40hour']=df['hours_per_week'].apply(lambda x: 1 if x <40 and x>=30 else 0 )
    df['50hour']=df['hours_per_week'].apply(lambda x: 1 if x <50 and x>=40 else 0 )
    df['60hour']=df['hours_per_week'].apply(lambda x: 1 if x>=50 else 0 )
    
    train=df[['age','power_age','fnlwgt','capital_gain','capital_loss','10hour','2030hour','40hour','50hour','60hour']]
    train=pd.concat([income,train,workclass,education,martial_status,occupation,relationship,race,sex,native],axis=1)
    for i in train.columns:
        train[i]=train[i].astype(np.float64)
    return(train)    

train=train_input(sys.argv[1])
train['fnlwgt']=train['fnlwgt'].apply(lambda x: (x-12285)/(1484705-12285))
train['capital_gain']=train['capital_gain'].apply(lambda x: (x)/99999)
train['capital_loss']=train['capital_loss'].apply(lambda x: (x)/4356)
train['age']=train['age'].apply(lambda x: (x-17)/73)
train['power_age']=train['power_age'].apply(lambda x: (x-289)/5329)     
x=np.matrix(train.drop(train.columns[0],axis=1))
y=np.matrix(train.iloc[:,0]).T  

def test_input(file):
    tf=pd.read_csv(file,encoding='BIG5')   
    worktest=pd.get_dummies(tf['workclass'])
    education=pd.get_dummies(tf['education'])
    education['ungraduate']=education[' 10th']+education[' 11th']+education[' 12th']+education[' HS-grad']
    education['junior']=education[' 1st-4th']+education[' 5th-6th']+education[' 7th-8th']+education[' 9th']
    education=education.drop([' 10th',' 11th',' 12th',' HS-grad',' 1st-4th',' 5th-6th',' 7th-8th',' 9th'],axis=1)
    martial_test=pd.get_dummies(tf['marital_status'])
    occupat_test=pd.get_dummies(tf['occupation'])
    relation_test=pd.get_dummies(tf['relationship'])
    race_test=pd.get_dummies(tf['race'])
    sex_test=pd.get_dummies(tf['sex'])
    native=pd.get_dummies(tf['native_country'])
    native.insert(0,'unknown_native',native.iloc[:,0])
    native=native.drop(native.columns[1],axis=1)
    native['poor']=native[' Cambodia']+native[' China']+native[' Columbia']+native[' Cuba']+native[' Dominican-Republic']+native[' Ecuador']+native[' El-Salvador']+native[' Guatemala']+native[' Haiti']+native[' Honduras']+native[' India']+native[' Iran']+native[' Jamaica']+native[' Laos']+native[' Mexico']+native[' Peru']+native[' Philippines']+native[' Thailand']+native[' Vietnam']+native[' Yugoslavia']+native[' Nicaragua']
    native['middle']=native[' Greece']+native[' Portugal']+native[' Trinadad&Tobago']+native[' Poland']+native[' South']+native[' Hungary']+native[' Taiwan']+native[' Outlying-US(Guam-USVI-etc)']+native[' Scotland']+native[' Puerto-Rico']+native[' England']+native[' France']+native[' Italy']+native[' Japan']+native[' Ireland']+native[' Germany']+native[' Hong']+native[' Canada']
    native['USA']=native[' United-States']
    native=pd.concat([native['USA'],native['middle'],native['poor']],axis=1)
    tf['power_age']=tf['age'].pow(2)
    tf['10hour']=tf['hours_per_week'].apply(lambda x: 1 if x <10 else 0 )
    tf['2030hour']=tf['hours_per_week'].apply(lambda x: 1 if x <30 and x>=10 else 0 )
    tf['40hour']=tf['hours_per_week'].apply(lambda x: 1 if x <40 and x>=30 else 0 )
    tf['50hour']=tf['hours_per_week'].apply(lambda x: 1 if x <50 and x>=40 else 0 )
    tf['60hour']=tf['hours_per_week'].apply(lambda x: 1 if x>=50 else 0 )
    test=tf[['age','power_age','fnlwgt','capital_gain','capital_loss','10hour','2030hour','40hour','50hour','60hour']]
    test=pd.concat([test,worktest,education,martial_test,occupat_test,relation_test,race_test,sex_test,native],axis=1)
    for i in test.columns:
        test[i]=test[i].astype(np.float64)
    return(test)  
    
test=test_input(sys.argv[2])
test['fnlwgt']=test['fnlwgt'].apply(lambda x: (x-12285)/(1484705-12285))
test['capital_gain']=test['capital_gain'].apply(lambda x: (x)/99999)
test['capital_loss']=test['capital_loss'].apply(lambda x: (x)/4356)
test['age']=test['age'].apply(lambda x: (x-17)/73)
test['power_age']=test['power_age'].apply(lambda x: (x-289)/5329)   
x_test=np.matrix(test)    


    
beta=logit(y,x,0.1,1000)
y_hat=predict(x_test,beta).astype(int)

test['label']=y_hat
y_hat=test.iloc[:,-1]
y_hat.index = list(range(1,16282))
y_hat.index = y_hat.index.set_names('id')
y_hat = y_hat.to_frame()
y_hat.to_csv(sys.argv[3])   
    

    
    