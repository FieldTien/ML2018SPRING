import numpy as np 
import gensim
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers
import re
import sys
f=open(sys.argv[2],'r')
lines1 = f.readlines()
for i in range(len(lines1)):

    lines1[i]=lines1[i].replace('[{}|^$&*\[\]_-<=>"@:;~`<>/-]','')
    lines1[i]=lines1[i].lower()
    lines1[i]=lines1[i].split()
      






f=open(sys.argv[1],'r')
lines = f.readlines()
label=np.zeros((len(lines)))
for i in range(len(lines)):
    lines[i]=lines[i].replace("\n",'')
    lines[i]=lines[i].replace('+++$+++','')
#    lines[i]=lines[i].replace(',','')
#    lines[i]=lines[i].replace('?','')
#    lines[i]=lines[i].replace('.','')
#    lines[i]=lines[i].replace('!','')
    
#    lines[i]=lines[i].replace(',','')
    lines[i]=lines[i].replace('[{}|^$&*\[\]_-<=>"@:;~`<>/-]','')
    

    lines[i]=lines[i].lower()
    lines[i]=lines[i].split()
    label[i]=lines[i][0]
    del lines[i][0]

lines1=lines1+lines




model = gensim.models.Word2Vec(lines1, min_count=3,size=100)
model.save('w2vector')

for i in range(len(lines)):
    lines[i]=[x for x in lines[i] if x in model]

word=np.zeros((200000,40,100))
for i in range(200000):
    if len(lines[i]) > 0: 
        say_vector = model[lines[i]]
        word[i,range(say_vector.shape[0]),:]=say_vector

model = Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=(40,100)))


model.add(LSTM(64))



model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.1)))
model.add(Dropout(0.4))


model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


del lines

del f



x_train, y_train = word[range(180000)], label[range(180000)]  
x_test , y_test  = word[range(180000,200000)], label[range(180000,200000)] 
del word
ckpt = ModelCheckpoint(filepath='val.{epoch:03d}-{acc:.5f}.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.fit(x_train, y_train, batch_size=256, epochs=30,validation_data=(x_test, y_test),callbacks=[ckpt])
#score = model.evaluate(x_test, y_test, batch_size=16)






