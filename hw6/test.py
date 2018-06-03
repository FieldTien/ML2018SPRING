import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Model,load_model
import pickle as pk
import sys
def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )


test=pd.read_csv(sys.argv[1])
tokenizer = pk.load(open('diction1.pickle', 'rb'))
a=test.MovieID.tolist()
a=[str(x) for x in a]

a=tokenizer.texts_to_sequences(a)
a=[a[x][0] for x in range(len(a))]
test.MovieID=a
model=load_model('bestmf.h5', custom_objects={'rmse': rmse})
model.summary()
y_hat=model.predict([test.UserID, test.MovieID])
y_hat=y_hat*1.116897661146206+3.5817120860388076
test['Rating']=y_hat

print(y_hat)

result=test['Rating']

result.index=test.TestDataID
result.index.name='TestDataID'

result.to_csv(sys.argv[2],header=True)
