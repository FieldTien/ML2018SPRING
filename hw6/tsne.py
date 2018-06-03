from keras.models import load_model
import pandas as pd 
import numpy as np 
import keras.backend as K
from keras.preprocessing.text import Tokenizer
import pickle as pk
from sklearn.manifold import TSNE
def rmse(y_true, y_pred): return K.sqrt( K.mean((y_pred - y_true)**2) )
model=load_model('bestmf.h5', custom_objects={'rmse': rmse})
model.summary() 
get_embbed=K.function([model.layers[1].input],[model.layers[5].output])

def data_pre(movies):
      #there are 6040 users and 3688 movies
    movie=pd.read_csv(movies,sep='::')
    a=[]
    for i in range(len(movie)):
        b=movie.Genres[i].split('|')
        a=a+b
    
    a=list(set(a))
    genres=pd.DataFrame(np.zeros((len(movie),len(a))))
    genres.columns=a
    for i in range(len(movie)):
        b=movie.Genres[i].split('|')
        for j in b:
            genres[j][i]=int(1)

    movie=pd.concat((movie,genres),axis=1)
    
    return(movie)

movie=data_pre('movies.csv')  
tokenizer = pk.load(open('diction1.pickle', 'rb'))
a=movie.movieID.tolist()
a=[str(x) for x in a]

a=tokenizer.texts_to_sequences(a)
a=[a[x][0] for x in range(len(a))]
movie.movieID=a

movie1=np.matrix(movie.movieID).T
movie_embbeded=get_embbed([movie1])[0]

print(movie)
a=np.zeros(len(movie))
for i in range(len(movie)):
    if movie.Animation[i]==1 or  movie["Children's"][i] ==1:
        a[i]=1
    elif movie.Horror[i]==1 or  movie.Thriller[i] ==1:
        a[i]=2
 
index=np.where(a>0)[0]

a=a[index]
movie_embbeded=movie_embbeded[index]
print(len(a))
print(movie_embbeded)

X_embedded = TSNE(n_components=2).fit_transform(movie_embbeded)
np.save('X_embedded.npy',X_embedded)
np.save('label.npy',a)
        

