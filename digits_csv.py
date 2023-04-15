from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np

digits = load_digits()
X,y = digits.data,digits.target.astype(int)

pca = PCA(n_components=2)
X_new = pca.fit_transform(X)


fmt = "%.5f,%.5f,%d"
stack = np.hstack([X_new,y.reshape((-1,1))])


np.savetxt("data/digits_2f.csv",stack,fmt=fmt,header="f1,f2,number",delimiter=",",comments="")

