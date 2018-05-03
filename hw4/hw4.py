import pandas as pd
import numpy as np
import sys

data=np.load(sys.argv[1])
q=np.zeros((data.shape[0]))
for i in range(data.shape[0]):
    if ((15 < data[i]) & (data[i] < 245)).sum()/784 < 0.15:
        q[i]=1
A=q
data=pd.read_csv(sys.argv[2])
result=np.zeros((1980000))
for i in range(1980000):
    if  A[data['image1_index'][i]]==A[data['image2_index'][i]]:
      result[i]=1
result=result.astype(int)    
data['Ans']=result
y_hat=data.iloc[:,-1]   
y_hat.index = y_hat.index.set_names('ID')
y_hat = y_hat.to_frame()   
y_hat.to_csv(sys.argv[3],header=True)
