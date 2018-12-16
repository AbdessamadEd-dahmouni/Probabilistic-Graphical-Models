from IPython import get_ipython
get_ipython().magic('matplotlib inline')
import random
import seaborn as sns
import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown
markers = ['*','+','.','1']

def fmt_output(symbol,n,shape,data):
    #for printing vectors and matrices
    L, C = shape
    G = [data] if n==1 else data
    S =r''
    for i in range(n):
        S +=r'$'+symbol+('_'+str(i+1) if n!=1 else '')
        S +=r'=\begin{bmatrix}' if (L!=1 or C!=1) else r'=' 
        M = [G[i]] if L==1 else G[i]
        for l in range(L):
            S += r'{:.2f}'.format(M[l]) if C==1 else (r'{:.2f} '+ r'& {:.2f}'*(C-1)).format(*M[l])
            if(l!= L-1):
                S+=r'\\'
        S +='\end{bmatrix}, \quad $ 'if (L!=1 or C!=1) else r', \quad $ ' 
    display(Markdown(S))

#%%
# ## 3.a.   K-means algorithm

class Kmeans:
    def __init__(self, data, K):
        self.K        = K
        self.data     = data 
        self.centers  = random.sample(list(data.values),K)                      # vector of centers mu_j
        self.previous = None                                                    # previous centers
        self.dist     = pd.DataFrame(index=data.index, columns=list(range(K)))  # matrix of distances ||x_i-mu_j||**2
        self.cluster  = None                                                    # index of closest center to x_i 
        self.iters    = 0                                                       # number of iterations
        self.distortion = None                                                  # the value of the distortion function J(z,mu)
    def update(self):
        self.previous = self.centers
        for j in range(self.K):
            self.dist[j] = (self.data - self.centers[j]).pow(2).sum(axis=1)
        self.cluster = self.dist.idxmin(axis=1)
        self.centers = [ np.array(self.data.loc[self.cluster == j].mean()) for j in range(self.K)]
       
    def fit(self):
        if self.iters == 0 : 
            self.update()
            self.iters += 1
        while sum(norm(self.centers[j]-self.previous[j]) for j in range(self.K))!=0:
            self.update()
            self.iters += 1
        self.distortion = self.dist.min(axis=1).sum()
    
    def predict(self,test):
        data = test.copy()
        dist = pd.DataFrame(index=data.index, columns=list(range(self.K)))
        for j in range(self.K):
            dist[j] = (data - self.centers[j]).pow(2).sum(1)
        data['state'] = dist.idxmin(axis=1)+1
        return data


#%%


# running K-means on the training data
if __name__=='__main__':
    # train and test datasets
    train = pd.read_csv('Data/EMGaussian.data',sep=" ", names=["x","y"])
    test  = pd.read_csv('Data/EMGaussian.test',sep=" ", names=["x","y"])
    # number of Clusters 
    K = 4
    model = Kmeans(train,K)
    model.fit()
    print('K-means Iterations: '+ str(model.iters))
    # cluster predictions on the training and test data
    clusters_train = model.predict(train)
    clusters_test  = model.predict(test)
    clusters_train['type'] = 'train'
    clusters_test['type']  = 'test'
    for_sns = pd.concat([clusters_train,clusters_test],axis=0)

    #plots
    g = sns.lmplot(data=for_sns, x='x', y='y', col='type', hue='state',markers=markers[:K], fit_reg=False)
    centers = list(zip(*model.centers))
    xRange = (for_sns['x'].min(), for_sns['x'].max())
    yRange = (for_sns['y'].min(), for_sns['y'].max())
    x = np.linspace(*xRange, 200)
    y = np.linspace(*yRange, 200)
    x, y = np.meshgrid(x, y)
    M = np.zeros((K,*x.shape))
    for j in range(K):
        M[j] = (x - model.centers[j][0])**2+(y - model.centers[j][1])**2
    M = np.sort(M, axis=0)
    for ax in g.axes[0]:
        plt.gca().set_aspect('equal')
        ax.scatter(*centers, color='black', marker='x');
        ax.contour(x, y, M[1]-M[0], [1], colors='k', linewidths=0.8);

    # trying multiple iterations
    c_data = pd.DataFrame(columns=['x','y'])
    N_inits = 10
    distortions = []
    iters_list  = []
    for i in range(N_inits):
        model = Kmeans(train,K)
        model.fit()
        c_data = c_data.append(pd.DataFrame(model.centers,columns=['x','y']))
        distortions.append(model.distortion)
        iters_list.append(model.iters)

    sns.lmplot(data=c_data, x='x', y='y', markers='x', fit_reg=False);
    plt.gca().set_title('Centers for several random initializations')
    plt.figure();plt.plot(distortions);plt.xlabel('Iteration');plt.ylabel('Distortion');
    plt.figure();plt.bar(list(range(N_inits)),iters_list);
    plt.xlabel('Number of the initialization');plt.ylabel('Number of iterations');

#%%
# ## 3.b.   EM algorithm, isotropic case


class EM_algorithm_iso:
    def __init__(self,data,K):
        model = Kmeans(data,K) 
        model.fit()
        self.K       = K
        self.d       = len(data.columns)
        self.data    = data
        self.tau_ji  = 1.0 - model.dist.gt(model.dist.min(axis=1),axis=0)   # matrix of tau_ji = P(z_i = j|x_i)
        self.p_j     = self.tau_ji.mean(axis=0)                             # vector of p_j = P(z=j)
        self.centers = model.centers              # vector of centers mu_j
        self.dist    = model.dist                 # matrix of distances ||x_i - mu_j||**2
        self.vars    = np.repeat(100.0,K)         # vector of variances for each matrix Sigma_j = var_j * I_2
        self.iters   = 0                          # number of iterations
        self.old_ll  = 0.0                        # old value of the log likelihood
        self.new_ll  = 0.0                        # new value of the log likelihood

    def E_step(self):
        self.tau_ji = self.p_j*np.exp(-0.5*self.dist/self.vars)/(2*np.pi*self.vars)
        self.tau_ji = self.tau_ji.div(self.tau_ji.sum(axis=1), axis=0)
        
    def M_step(self):
        self.p_j = self.tau_ji.mean(axis=0)
        for j in range(self.K):
            self.centers[j] = np.average(self.data, weights=self.tau_ji[j], axis=0)
            self.dist[j]    = (self.data-self.centers[j]).pow(2).sum(axis=1)
            self.vars[j]    = np.average(self.dist[j], weights=self.tau_ji[j], axis=0)/float(self.d)
            
    def update_ll(self):
        self.old_ll = self.new_ll
        p_xi = (self.p_j*np.exp(-0.5*self.dist/self.vars)/(2*np.pi*self.vars)).sum(axis=1)
        self.new_ll = np.sum(np.log(p_xi))
        return self.new_ll
        
    def update(self):
            self.E_step()
            self.M_step()
            self.update_ll() 
            
    def fit(self,tol=1e-6, max_iters=1000, verbose=False):
        if self.iters == 0 :
            self.update()
            self.iters += 1
            if verbose:
                print('Iteration: 1, log likelihood: {:.4f}'.format(self.new_ll))
        while(np.abs(self.new_ll-self.old_ll)>tol and self.iters<max_iters):
            self.update()
            self.iters += 1
            if verbose and self.iters%10 == 0 :
                print('Iteration: {:d}, log likelihood: {:.4f}'.format(self.iters, self.new_ll))
        if verbose:
            print('Last iteration: {:d}, log likelihood: {:.4f}'.format(self.iters, self.new_ll))
        
    def ll(self, data):
        dist = pd.DataFrame(index=data.index, columns=list(range(self.K)))
        for j in range(self.K):
            dist[j] = (data - self.centers[j]).pow(2).sum(1)
        p_xi = (self.p_j*np.exp(-0.5*dist/self.vars)/(2*np.pi*self.vars)).sum(axis=1)
        return np.sum(np.log(p_xi)), np.mean(np.log(p_xi))
        
    def predict(self,test):
        data = test.copy()
        dist = pd.DataFrame(index=data.index, columns=list(range(self.K)))
        for j in range(self.K):
            dist[j] = (data - self.centers[j]).pow(2).sum(1)
        tau_ji  = self.p_j*np.exp(-0.5*dist/self.vars)/(2*np.pi*self.vars)  
        # here tau_ji not normalized since it's only used to find the maximum
        data['state'] = tau_ji.idxmax(axis=1)+1
        return data
#%%
if __name__=='__main__':
    # running isotropic EM on the training data
    EM_iso = EM_algorithm_iso(train,K)
    EM_iso.fit(tol=1e-6);
    # cluster predictions on the training and test data
    clusters_train = EM_iso.predict(train)
    clusters_test  = EM_iso.predict(test)
    clusters_train['type'] = 'train'
    clusters_test['type']  = 'test'
    for_sns = pd.concat([clusters_train,clusters_test],axis=0)

    #plots
    pct = 0.9 # percentage of distribution inside the circle
    g = sns.lmplot(data=for_sns, x='x', y='y', col='type', hue='state',markers=markers[:K], fit_reg=False)
    centers = list(zip(*EM_iso.centers))
    for ax in g.axes[0]:
        plt.gca().set_aspect('equal')
        ax.scatter(*centers, color = 'black', marker ='x');
        xRange = (for_sns['x'].min(), 1.5*for_sns['x'].max())
        yRange = (for_sns['y'].min(), 1.5*for_sns['y'].max())
        x = np.linspace(*xRange, 100)
        y = np.linspace(*yRange, 100)
        x, y = np.meshgrid(x, y)
        for j in range(K):
            ax.contour(x, y, (x-EM_iso.centers[j][0])**2+(y-EM_iso.centers[j][1])**2,
                       [-2*EM_iso.vars[j]*np.log(1-pct)], colors='k', linewidths=0.8)

#%%
# ## 3.c.   EM algorithm, general case



class EM_algorithm:
    def __init__(self,data,K):
        model = Kmeans(data,K) 
        model.fit()
        self.K       = K
        self.data    = data
        self.tau_ji  = 1.0 - model.dist.gt(model.dist.min(axis=1),axis=0)
        self.p_j     = self.tau_ji.mean(axis=0) 
        self.centers = model.centers
        self.dist    = pd.DataFrame(index=data.index, columns=list(range(K)))
        self.sigmas  = [np.diag([10.0,10.0]) for j in range(K)]                # list of covariance matrices Sigma_j
        self.dets    = np.repeat(100.0,K)
        self.iters   = 0
        self.old_ll  = 0.0
        self.new_ll  = 0.0

    def E_step(self):
        for j in range(self.K):
            delta        = (self.data-self.centers[j]).values
            self.dist[j] = np.dot(delta,np.dot(np.linalg.inv(self.sigmas[j]),delta.T)).diagonal()
        self.tau_ji = self.p_j*np.exp(-0.5*self.dist)/(2*np.pi*np.sqrt(self.dets))
        self.tau_ji = self.tau_ji.div(self.tau_ji.sum(axis=1), axis=0)
        
    def M_step(self):
        self.p_j = self.tau_ji.mean(axis=0)
        for j in range(self.K):
            self.centers[j] = np.average(self.data, weights=self.tau_ji[j], axis=0)
            delta           = (self.data-self.centers[j]).values
            self.sigmas[j]  = np.dot(delta.T,self.tau_ji[j].values[:,None]*delta)/self.tau_ji[j].sum()
            self.dets[j]    = np.linalg.det(self.sigmas[j])
            
    def update_ll(self):
        self.old_ll = self.new_ll
        p_xi = (self.p_j*np.exp(-0.5*self.dist)/(2*np.pi*np.sqrt(self.dets))).sum(axis=1)
        self.new_ll = np.sum(np.log(p_xi))
        
    def update(self):
            self.E_step()
            self.M_step()
            self.update_ll() 
            
    def fit(self,tol=1e-6, max_iters=1000, verbose=False):
        if self.iters == 0 :
            self.update()
            self.iters += 1
            if verbose:
                print('Iteration: 01, log likelihood: {:.4f}'.format(self.new_ll))
        while(np.abs(self.new_ll-self.old_ll)>tol and self.iters<max_iters):
            self.update()
            self.iters += 1
            if verbose and self.iters%10 == 0 :
                print('Iteration: {:d}, log likelihood: {:.4f}'.format(self.iters, self.new_ll))
        if verbose:
            print('Last iteration: {:d}, log likelihood: {:.4f}'.format(self.iters, self.new_ll))

    def ll(self, data):
        dist = pd.DataFrame(index=data.index, columns=list(range(self.K)))
        for j in range(self.K):
            delta   = (data-self.centers[j]).values
            dist[j] = np.dot(delta,np.dot(np.linalg.inv(self.sigmas[j]),delta.T)).diagonal()
        p_xi = (self.p_j*np.exp(-0.5*dist)/(2*np.pi*np.sqrt(self.dets))).sum(axis=1)
        return np.sum(np.log(p_xi)), np.mean(np.log(p_xi))
    
    def predict(self,test):
        data = test.copy()
        dist = pd.DataFrame(index=data.index, columns=list(range(self.K)))
        for j in range(self.K):
            delta   = (data-self.centers[j]).values
            dist[j] = np.dot(delta,np.dot(np.linalg.inv(self.sigmas[j]),delta.T)).diagonal()
        tau_ji  = self.p_j*np.exp(-0.5*dist)/(2*np.pi*np.sqrt(self.dets)) #proportional
        data['state'] = tau_ji.idxmax(axis=1)+1
        return data


#%%

if __name__=='__main__':
    # running general EM on the training data
    EM = EM_algorithm(train,K)
    EM.fit(tol=1e-6,verbose=True);
    # cluster predictions on the training and test data
    clusters_train = EM.predict(train)
    clusters_test  = EM.predict(test)
    clusters_train['type'] = 'train'
    clusters_test['type']  = 'test'
    for_sns = pd.concat([clusters_train,clusters_test],axis=0)
    
    #plots
    # percentage of distribution inside the ellipse
    pct = 0.9
    g = sns.lmplot(data=for_sns, x='x', y='y', col='type', hue='state',markers=markers[:K], fit_reg=False)
    centers = list(zip(*EM.centers))
    for ax in g.axes[0]:
        plt.gca().set_aspect('equal')
        ax.scatter(*centers, color='black', marker='x');
        xRange = (for_sns['x'].min(), for_sns['x'].max())
        yRange = (for_sns['y'].min(), for_sns['y'].max())
        x = np.linspace(*xRange, 100)
        y = np.linspace(*yRange, 100)
        x, y = np.meshgrid(x, y)
        for j in range(K):
            Inv = np.linalg.inv(EM.sigmas[j])
            a, b, c = Inv[0][0], Inv[1][0]+ Inv[0][1], Inv[1][1]
            ax.contour(x, y, a*(x-EM.centers[j][0])**2 + b*(x-EM.centers[j][0])*(y-EM.centers[j][1])+c*(y-EM.centers[j][1])**2
                       ,[-2*np.log(1-pct)], colors = 'k', linewidths=0.8)
    
    # Comparing log likelihoods:
    print('Isotropic EM:')
    print('   - Training set: {:.3f}'.format(EM_iso.ll(train)[0]))
    print('   - Test set:     {:.3f}'.format(EM_iso.ll(test)[0]))
    print('General EM:')
    print('   - Training set: {:.3f}'.format(EM.ll(train)[0]))
    print('   - Test set:     {:.3f}'.format(EM.ll(test)[0]))

