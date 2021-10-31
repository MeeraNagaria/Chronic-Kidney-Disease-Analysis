import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def pca_preprocess(data,feature1,feature2):

    data.fillna(0)
    data_pca=data.replace(np.nan,0)
   
    x = data_pca.loc[:,feature1].values
 
    y = data_pca.loc[:,feature2].values
    
    x = StandardScaler().fit_transform(x)
    return x,y

def pca_analysis(x,y,component):
       x=pd.DataFrame(x)
       y=pd.DataFrame(y)
       pca = PCA(n_components=3)
       principalComponents = pca.fit_transform(x)
       principalDf = pd.DataFrame(data = principalComponents
                                  , columns = ['principal component 1', 'principal component 2','principal component 3'])
       
       finalDf = pd.concat([principalDf,y], axis = 1) #Used for plotting
       print(pca.explained_variance_ratio_)
       return print(abs( pca.components_ ))