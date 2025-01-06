import numpy as np
from random import randrange

# Defining Function
SL_Val = ""
def f(x):
    return x ** 3 - 5 * x - 9

# Implementing Secant line Method
def secant(x0, x1, e, N):
    global SL_Val
    print('\n\n*** SECANT LINE METHOD ***')
    step = 1
    condition = True
    while condition:
        if f(x0) == f(x1):
            print('Divide by zero error!')
            break
        x2 = x0 - (x1 - x0) * f(x0) / (f(x1) - f(x0))
        print('Iteration-%d, x2 = %0.6f and f(x2) = %0.6f' % (step, x2, f(x2)))
        x0 = x1
        x1 = x2
        step = step + 1
        if step > N:
            print('')
            break
        condition = abs(f(x2)) > e
    SL_Val = x2
    #print("Required farthest point is : ",int(SL_Val))
x0 = randrange(2,4)
x1 = randrange(4,5)
e = 0.000001
N = randrange(10)
# Converting x0 and e to float
x0 = float(x0)
x1 = float(x1)
e = float(e)
# Converting N to integer
N = int(N)
secant(x0, x1, e, N)
def rffc(X,D,k):
    """
    X: input vectors (n_samples by dimensionality)
    D: distance matrix (n_samples by n_samples)
    k: number of centroids
    out: indices of centroids
    """
    n=X.shape[0]
    visited=[]
    #the initial point, the farthest points are chosen
    i=np.int32(SL_Val)
    visited.append(i)
    while len(visited)<k:
        dist=np.mean([D[i] for i in visited],0)
        for i in np.argsort(dist)[::-1]:
            if i not in visited:
                visited.append(i)
                break
    return np.array(visited)
import matplotlib.pyplot as plt
n=300
e=0.1
mu1=np.array([-2.,0.])
mu2=np.array([2.,0.])
mu3=np.array([0.,2.])
mu=np.array([mu1,mu2,mu3])
x1=np.random.multivariate_normal(mu1,e*np.identity(2),n//2)
x2=np.random.multivariate_normal(mu2,e*np.identity(2),n//2)
#x3=np.random.multivariate_normal(mu3,e*np.identity(2),n//3)
X=np.r_[x1,x2]
y=np.concatenate([np.repeat(0,int(n/2)),np.repeat(1,int(n/2))])
X2=np.c_[np.sum(X**2,1)]
D=X2+X2.T-2*X.dot(X.T)
centroid_idx=rffc(X,D,2)
centroids=X[centroid_idx]
colors=plt.cm.Paired(np.linspace(0, 1, len(np.unique(y))))
plt.scatter(X[:,0],X[:,1],color=colors[y])
plt.scatter(centroids[:,0],centroids[:,1],color="black",s=50)
plt.show()
