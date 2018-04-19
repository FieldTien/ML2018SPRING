# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 00:15:17 2018

@author: Field Tien
"""
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import np_utils
import matplotlib.pyplot as plt


data=pd.read_csv('train.csv')
label = np.array(data['label'])
img_data = data['feature'].apply(lambda x: x.split(' '))
train=np.zeros(len(label)*48*48).reshape(len(label),48,48,1)
for i in range(len(label)):
    train[i,:,:,:]=np.array(img_data[i]).reshape(48,48,1)
train=train/255
y=np_utils.to_categorical(label, num_classes=7)

X_test, y_test = train[range(23000,28709)], y[range(23000,28709)]


model = load_model('best.h5')
predict = model.predict(X_test)
predict = np.argmax(predict, 1)
y_test = np.argmax(y_test, 1)
CM=confusion_matrix(y_test, predict)
CM = CM.astype('float') / CM.sum(axis=1)[:, np.newaxis]

listClasses=['angry','disgust','fear','happy','sad','suprise','neautral']
confusionmatrix=CM
cmap = plt.cm.jet
confusionmatrix = confusionmatrix.astype("float") / confusionmatrix.sum(axis=1)[:, np.newaxis]
a=plt.figure(figsize=(8, 6))
plt.imshow(confusionmatrix, interpolation="nearest", cmap='Greens')
plt.title('confusionmatrix')
plt.colorbar()
tick_marks = np.arange(len(listClasses))
plt.xticks(tick_marks, listClasses, rotation=45)
plt.yticks(tick_marks, listClasses)

thresh = confusionmatrix.max() / 2.
for i, j in itertools.product(range(confusionmatrix.shape[0]), range(confusionmatrix.shape[1])):
    plt.text(j, i, "{:.2f}".format(confusionmatrix[i, j]), horizontalalignment="center",
             color="white" if confusionmatrix[i, j] > thresh else "black")
plt.tight_layout() 
plt.ylabel("True label")
plt.xlabel("Predicted label")
a.savefig('q3.png')





















CLASSES=range(7)
sb.set(font_scale=1.4)
Com = sb.heatmap(CM, annot=True, annot_kws={"size": 16 }, fmt='d', cmap='YlGnBu')
Com.set_xticklabels(label)
Com.set_yticklabels(label)
plt.yticks(rotation=0)
plt.xlabel("Predict")
plt.ylabel("True")
Com.get_figure().savefig(model_name[:-3] + "_cm.png")

cmap = plt.cm.jet
CM = CM.astype("float") / CM.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
plt.imshow(confusionmatrix, interpolation="nearest", cmap=cmap)
plt.title(title)
    plt.colorbar()
    
classes=range(6)    
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
    horizontalalignment="center",
    color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
 

fig, ax = plot_confusion_matrix(conf_mat=CM)
plt.show()   