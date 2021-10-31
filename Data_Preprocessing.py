from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def data_checknull(data):
    data = data.replace('?', np.nan) 
    print (data.isna().sum() )
    return data

def data_drop_columns(data,column_numbers):
    data = data.drop(columns=column_numbers)
    return data
    
def data_replace(data): 
   data = data.replace({'normal':0,'abnormal':1})
   data1 = data.replace({'notpresent':0,'present':1})
   data2 = data1.replace({'no':0,'yes':1})
   data3 = data2.replace({'good':0,'poor':1})
   return data3

def data_transform(data):
    scaler = StandardScaler() 
    data_scaled = scaler.fit_transform(data)
    data_transform = pd.DataFrame(data_scaled)
    return data_transform

def data_correlation (data):
    corrMatrix = data.corr()
    sn.heatmap(corrMatrix, annot=False)
    plt.show()
    
    
    
    
    


