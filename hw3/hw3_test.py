import sys
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import Activation
from keras.layers.advanced_activations import *
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import load_model
model = load_model('best.h5')
data=pd.read_csv(sys.argv[1])
ID = np.array(data['id'])
img_data = data['feature'].apply(lambda x: x.split(' '))
train=np.zeros(len(ID)*48*48).reshape(len(ID),48,48,1)
for i in range(len(ID)):
    train[i,:,:,:]=np.array(img_data[i]).reshape(48,48,1)
train=train/255
y_hat=model.predict(train)
y_hat=np.argmax(y_hat, axis=1)
data['label']=y_hat
y_hat=data.iloc[:,-1]
y_hat.index = data['id']
y_hat.index = y_hat.index.set_names('id')
y_hat = y_hat.to_frame()
y_hat.to_csv(sys.argv[2])

