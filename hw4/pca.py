from skimage import io
import numpy as np 
import sys

a=np.zeros((1080000,415))
filefold=sys.argv[1]
for i in range(415):
    filename=filefold+'/%s.jpg'%i
    img = io.imread(filename)
    img=img.flatten()
    a[:,i]=img
X=a.astype(int) 
X_mean=np.mean(X, axis=1)

demean=np.zeros((1080000,415))

for i in range(415):
    demean[:,i]=X[:,i]-X_mean
    
U, s, V = np.linalg.svd(demean, full_matrices=False)


a=sys.argv[2]
a.split('.jpg')
i=int(a[0])
weight=np.dot(demean.T,U)
reconstrutc=X_mean+weight[i,0]*U[:,0]+weight[i,1]*U[:,1]+weight[i,2]*U[:,2]+weight[i,3]*U[:,3]
reconstrutc=reconstrutc.reshape((600,600,3))
reconstrutc = (reconstrutc-np.min(reconstrutc))
reconstrutc = reconstrutc/np.max(reconstrutc)
reconstrutc=(reconstrutc*255).astype(np.uint8)
io.imsave('reconstruction.jpg',reconstrutc)



