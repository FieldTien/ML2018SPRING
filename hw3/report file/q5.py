# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 20:55:05 2018

@author: Field Tien
"""
import numpy as np
import pandas as pd
from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import time
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x
data=pd.read_csv('train.csv')
label = np.array(data['label'])
img_data = data['feature'].apply(lambda x: x.split(' '))
train=np.zeros(len(label)*48*48).reshape(len(label),48,48,1)
for i in range(len(label)):
    train[i,:,:,:]=np.array(img_data[i]).reshape(48,48,1)

pic=train[0].reshape(1,48,48,1)
img_width =48
img_height = 48

model = load_model('best.h5')
input_img = model.input
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_name = "conv2d_1"
random_img = np.random.random((1, 48, 48, 1))
layer_output = layer_dict[layer_name].output
nb_filter = 50
fig = plt.figure(figsize=(14, 8))
for i in range(nb_filter):
    ax = fig.add_subplot(nb_filter/10, 10, i+1)
    loss = K.mean(layer_output[:, :, :,i])
    grads = K.gradients(loss, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([input_img], [loss, grads])
    input_img_data = np.array(random_img)
    step = 1
    for j in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img= input_img_data[0].reshape(48, 48)
    img = deprocess_image(img)
    ax.imshow(img, cmap='Greens')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel('filter {}'.format(i))
    plt.tight_layout()


fn=K.function([input_img, K.learning_phase()], [layer_dict['conv2d_1'].output])
im=fn([pic, 0])
im=np.array(im).reshape(48,48,50)
fig.savefig('q5_1.png')


fig2 = plt.figure(figsize=(14, 8))
        #nb_filter = im[0].shape[3]

for i in range(nb_filter):
    ax = fig2.add_subplot(nb_filter/10, 10, i+1)
    ax.imshow(im[0:48,0:48,i], cmap='Greens')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.xlabel('filter {}'.format(i))
    plt.tight_layout()
fig2.savefig('q5_2.png')    
true=pic.reshape(48,48)
fig3= plt.figure(figsize=(14, 8))
plt.imshow(true,cmap='gray') 
fig3.savefig('q5_3.png')    
    