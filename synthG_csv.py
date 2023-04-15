import numpy as np

def generate_data(k=3, dim=2, points_range=[100,200], lim=[-15, 15],seed=45):
    np.random.seed(seed)
    mean = np.random.rand(k, dim)*(lim[1]-lim[0]) + lim[0]
    x = []
    y = []
    for i in range(k):
        cov = np.random.rand(dim, 10)
        cov = np.matmul(cov, cov.T) 
        points = np.random.randint(*points_range)
        _x = np.random.multivariate_normal(mean[i], cov, points)
        x += list(_x)
        y += [i]* points
    x = np.array(x)
    y = np.array(y) 
    return x,y
X,y = generate_data(7)
fmt = "%.5f,%.5f,%d"
stack = np.hstack([X,y.reshape((-1,1))])
np.savetxt("data/synth_7g.csv",stack,fmt=fmt,header="f1,f2,target",delimiter=",",comments="")

