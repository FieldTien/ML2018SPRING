import numpy as np 
import gensim
from keras.models import load_model
from gensim.models import Word2Vec
import pandas as pd
import sys


model = Word2Vec.load('w2vector')
f=open(sys.argv[1],'r')
lines = f.readlines()
idd=np.zeros((len(lines)-1))
for i in range(1,(len(lines))):
    lines[i]=lines[i].replace("\n",'')
    lines[i]=lines[i].replace('[{}|^$&*\[\]_-<=>"@:;~`<>/-]','')
    lines[i]=lines[i].lower()
    lines[i]=lines[i].split(',',1)
    idd[i-1]=int(lines[i][0])
    lines[i]=lines[i][1].split()
    
    lines[i]=[x for x in lines[i] if x in model]

del lines[0]




word=np.zeros((200000,40,100))

for i in range(200000):
    if len(lines[i]) > 0:
        say_vector = model[lines[i]]
        word[i,range(say_vector.shape[0]),:]=say_vector
    
    
model = load_model('best.h5')
y_hat=model.predict(word)
label=np.zeros((200000))
for i in range(200000):
    if y_hat[i,0] >= 0.5:
        label[i]=1
label=label.astype(int)  
idd=idd.astype(int)
label=np.matrix(label).T
idd=np.matrix(idd).T

result=pd.DataFrame(idd)
result.columns=['id']
result['label']=label


result.to_csv(sys.argv[2],index=False)

