import numpy as np
import pandas as pd
import keras.backend as K
from sklearn.model_selection import train_test_split
import keras
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Input ,Embedding ,Dot ,Add,Flatten
from sklearn.model_selection import train_test_split
import keras
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.regularizers import l2
from keras.preprocessing.text import Tokenizer
import pickle as pk
def rmse(y_true, y_pred): return K.sqrt( K.mean(((y_pred) - (y_true))**2) )
train=pd.read_csv("train .csv")  #there are 6040 users and 3688 movies
movie=pd.read_csv('movies.csv',sep='::')
a=pd.DataFrame(np.concatenate((train.MovieID,movie.movieID),axis=0))
a.columns=['s']
n_movies=len(a.s.unique())
a=a.s.tolist()
a=[str(x) for x in a]
tokenizer=Tokenizer()
tokenizer.fit_on_texts(a)
pk.dump(tokenizer, open('diction1.pickle', 'wb'))
name=list(movie.columns)
name[0]='MovieID'
movie.columns=name
#user=pd.read_csv('users.csv',sep='::')
#train1=pd.merge(train,movie,on='MovieID')
#train1=pd.merge(train1,user,on='UserID')
token=train.MovieID.tolist()
token=[str(x) for x in token]
token=tokenizer.texts_to_sequences(token)
print(len(token))
token=[token[x][0] for x in range(len(token))]
train.MovieID=token

print(train)
shuffle=np.random.permutation(len(train.UserID))
train.UserID=train.UserID[shuffle]
train.MovieID=train.MovieID[shuffle]
train.Rating=train.Rating[shuffle]


#print(np.mean(train.Rating))
#print(np.std(train.Rating))
train.Rating=(train.Rating-np.mean(train.Rating))/np.std(train.Rating)

n_user= len(train.UserID.unique())
EMB_DIM = 128

print(n_user,n_movies)








movie_input = Input(shape=[1])
movie_embedding = Embedding(n_movies, EMB_DIM)(movie_input)
movie_vec = Flatten(name='FlattenMovies')(movie_embedding)
movie_vec=Dropout(0.5)(movie_vec)
user_input = Input(shape=[1])
user_embedding = Embedding(n_user, EMB_DIM)(user_input)
user_vec = Flatten(name='FlattenUsers')(user_embedding)
user_vec=Dropout(0.5)(user_vec)
movie_bias=Embedding(n_movies, 1)(movie_input)
movie_bias=Flatten()(movie_bias)


user_bias=Embedding(n_user, 1)(user_input)
user_bias=Flatten()(user_bias)

prod=Dot(axes=-1)([user_vec,movie_vec])
prod=Add()([prod,user_bias,movie_bias])
model = Model([user_input, movie_input], prod)
model.compile(optimizer='adam', loss='mse', metrics=[rmse])
callbacks = []
callbacks.append(EarlyStopping(monitor='val_rmse', patience=20,verbose=1, mode='min'))
callbacks.append(ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True,mode='min'))
model.fit([train.UserID, train.MovieID], train.Rating,epochs=200, batch_size=512, validation_split=0.05, callbacks=callbacks)

