# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 01:28:42 2018

@author: Field Tien
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
from keras.utils import plot_model
from keras.models import load_model
from keras import backend as K

data=pd.read_csv('test.csv')
ID = np.array(data['id'])
img_data = data['feature'].apply(lambda x: x.split(' '))
train=np.zeros(len(ID)*48*48).reshape(len(ID),48,48,1)
for i in range(len(ID)):
    train[i,:,:,:]=np.array(img_data[i]).reshape(48,48,1)
train=train/255
model = load_model('best.h5')
inputImage = model.input
arrayProbability = model.predict(train[100].reshape(1, 48, 48, 1))
arrayPredictLabel = arrayProbability.argmax(axis=-1)
tensorTarget = model.output[:, arrayPredictLabel]
tensorGradients = K.gradients(tensorTarget, inputImage)[0]
fn = K.function([inputImage, K.learning_phase()], [tensorGradients])
arrayGradients = fn([train[100].reshape(1, 48, 48, 1), 0])[0].reshape(48, 48, -1)
arrayGradients = np.max(np.abs(arrayGradients), axis=-1, keepdims=True)
arrayGradients = (arrayGradients - np.mean(arrayGradients)) / (np.std(arrayGradients) + 1e-5)
arrayGradients *= 0.1
arrayGradients += 0.5
arrayGradients = np.clip(arrayGradients, 0, 1)

arrayHeatMap = arrayGradients.reshape(48, 48)
true=train[100].reshape(48, 48)

new=np.zeros((48, 48))
for i in range(48):
    for j in range(48):
        if arrayHeatMap[i,j] > 0.5:
            new[i,j] =true[i,j]
            
            
arrayProbability = model.predict(train[10].reshape(1, 48, 48, 1))
arrayPredictLabel = arrayProbability.argmax(axis=-1)
tensorTarget = model.output[:, arrayPredictLabel]
tensorGradients = K.gradients(tensorTarget, inputImage)[0]
fn = K.function([inputImage, K.learning_phase()], [tensorGradients])
arrayGradients = fn([train[10].reshape(1, 48, 48, 1), 0])[0].reshape(48, 48, -1)
arrayGradients = np.max(np.abs(arrayGradients), axis=-1, keepdims=True)
arrayGradients = (arrayGradients - np.mean(arrayGradients)) / (np.std(arrayGradients) + 1e-5)
arrayGradients *= 0.1
arrayGradients += 0.5
arrayGradients = np.clip(arrayGradients, 0, 1)

arrayHeatMap2 = arrayGradients.reshape(48, 48)
true2=train[10].reshape(48, 48)

new2=np.zeros((48, 48))
for i in range(48):
    for j in range(48):
        if arrayHeatMap2[i,j] > 0.5:
            new2[i,j] =true2[i,j]

arrayProbability = model.predict(train[5].reshape(1, 48, 48, 1))
arrayPredictLabel = arrayProbability.argmax(axis=-1)
tensorTarget = model.output[:, arrayPredictLabel]
tensorGradients = K.gradients(tensorTarget, inputImage)[0]
fn = K.function([inputImage, K.learning_phase()], [tensorGradients])
arrayGradients = fn([train[5].reshape(1, 48, 48, 1), 0])[0].reshape(48, 48, -1)
arrayGradients = np.max(np.abs(arrayGradients), axis=-1, keepdims=True)
arrayGradients = (arrayGradients - np.mean(arrayGradients)) / (np.std(arrayGradients) + 1e-5)
arrayGradients *= 0.1
arrayGradients += 0.5
arrayGradients = np.clip(arrayGradients, 0, 1)

arrayHeatMap3 = arrayGradients.reshape(48, 48)
true3=train[5].reshape(48, 48)

new3=np.zeros((48, 48))
for i in range(48):
    for j in range(48):
        if arrayHeatMap3[i,j] > 0.5:
            new3[i,j] =true3[i,j]
            
q4=plt.figure(figsize=(8,8))  
plt.subplot(3,3,1)   
plt.title('true')  
plt.imshow(true,cmap='gray')    
plt.colorbar() 
plt.axis('off')
plt.subplot(3,3,2)   
plt.title('Saliency Map')   
plt.imshow(arrayHeatMap, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.axis('off')
plt.subplot(3,3,3)     
plt.title('Mask')   
plt.imshow(new, cmap='gray')
plt.colorbar()
plt.axis('off')
plt.subplot(3,3,4)   
plt.imshow(true2,cmap='gray')   
plt.colorbar()
plt.axis('off')  
plt.subplot(3,3,5)   
plt.imshow(arrayHeatMap2, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.axis('off')
plt.subplot(3,3,6)     
plt.imshow(new2, cmap='gray')
plt.colorbar()
plt.axis('off')
plt.subplot(3,3,7)   
plt.imshow(true3,cmap='gray')  
plt.colorbar()
plt.axis('off')   
plt.subplot(3,3,8)   
plt.imshow(arrayHeatMap3, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.axis('off')
plt.subplot(3,3,9)     
plt.imshow(new3, cmap='gray')
plt.colorbar()
plt.axis('off')
plt.show()   
q4.savefig('q4.png')