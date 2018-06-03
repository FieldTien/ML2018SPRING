import numpy as np 
import matplotlib.pyplot as plt
 

embed=np.load('X_embedded.npy')
label=np.load('label.npy')

child=np.where(label==1)[0]
animate=np.where(label==2)[0]

plt.scatter(embed[:,0],embed[:,1],c=label,s=3)

plt.colorbar()

plt.show()

