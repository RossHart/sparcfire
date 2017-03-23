from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from tabulate import tabulate

def pca_analysis(X,standardise=True,printout=True,plot=False):
    
    def score(X,pca,component=1):
        N_x, N_var = X.shape
        score = np.zeros(N_x)
        for v in range(N_var):
            score += X[:,v]*(pca.components_[component-1,v])
        return score
    
    n_x, n_var = X.shape
    if standardise is True:
        X_scaled = scale(X)
    else: 
        X_scaled = X.copy()
        
    pca = PCA().fit(X_scaled)
    n_components = len(pca.explained_variance_ratio_)
    if plot is True:
        fig = plt.figure(figsize=(5,3))
        _ = plt.plot(np.arange(n_components)+1,pca.explained_variance_ratio_,
                     color='k',lw=2,marker='o')
        plt.xlabel('component')
        plt.ylabel('ratio')

    correlations = np.zeros((n_var,n_components))
    for component in range(n_components):
        component_score = score(X,pca,component+1)
        for v in range(n_var):
            c_coefficient, _ = spearmanr(component_score,X_scaled[:,v])
            correlations[v,component] = np.round(c_coefficient,decimals=2)
    if printout is True:
        colnames = ['Variable']
        [colnames.append('PCA{}'.format(c+1)) for c in range(n_components)]
        print_array = np.chararray((n_var+1,n_components+1),itemsize=16,unicode=True)
        print_array[0,0] = 'Exp. var. ratio'
        print_array[1:,0] = ['rho{}'.format(v+1) for v in range(n_var)]
        print_array[0,1:] = np.round(pca.explained_variance_ratio_,decimals=2)
        print_array[1:,1:] = np.round(correlations,decimals=2)
        print_table = tabulate(print_array,headers=colnames,tablefmt='fancy_grid')
        print(print_table)

    return correlations