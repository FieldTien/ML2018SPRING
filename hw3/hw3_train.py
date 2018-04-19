import sys
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
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
from keras.optimizers import SGD

batch_size = 128
epochs = 500
data=pd.read_csv(sys.argv[1])
label = np.array(data['label'])
img_data = data['feature'].apply(lambda x: x.split(' '))
train=np.zeros(len(label)*48*48).reshape(len(label),48,48,1)
for i in range(len(label)):
    train[i,:,:,:]=np.array(img_data[i]).reshape(48,48,1)
train=train/255
y=np_utils.to_categorical(label, num_classes=7)

X_train, y_train = train[range(23000)], y[range(23000)]
X_test, y_test = train[range(23000,28709)], y[range(23000,28709)]

#make partition for validation
datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=[0.8, 1.2],
            shear_range=0.2,
            horizontal_flip=True)


#construct CNN
model = Sequential()
model.add(Conv2D(50, (3, 3), input_shape=(48, 48,1),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(200, (3, 3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(400, (3, 3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(400, (3, 3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(units = 500, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units = 300, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units = 200, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units = 7, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
ckpt = ModelCheckpoint(filepath='val.{epoch:03d}-{acc:.5f}.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
print('train_________')
model.fit_generator(
            datagen.flow(X_train, y_train, batch_size=batch_size), 
            steps_per_epoch=5*len(X_train)//batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[ckpt]
            
            )
print('\nTesting______')
loss, accuracy=model.evaluate(X_test,y_test)
print('\ntest loss:',loss)
print('\ntest accuracy:',accuracy)
