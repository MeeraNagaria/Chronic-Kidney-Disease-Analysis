import import_data as ID
import Data_Preprocessing as DP
import PCA_Analysis as PA
import Classification as CLA
import pandas as pd
from sklearn.model_selection import KFold

data_csv=pd.read_csv("C:/Users/meera/Downloads/device_failure.csv",encoding='cp1252')
#kf = KFold(n_splits=2)


data = ID.import_file()
print (data.head())


data = DP.data_checknull(data)
data_dropped = DP.data_drop_columns(data,['id','rbc','wbcc','rbcc'])
data_converted = DP.data_replace(data_dropped)
#data_transform = DP.data_transform(data_convert)

#PCA Analysis
#x,y= PA.pca_preprocess(data_converted, ['age','bp','sg','al','su','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','htn','dm','cad','appet','pe','ane'],['class'])
# x,y= PA.pca_preprocess(data_converted,['age','bp','sg','al','su' ,'bgr','bu','sc','sod','pot','hemo','pcv'],['class'])
# PCA_result=PA.pca_analysis(x,y,3)
class_count_0, class_count_1 = data_csv['failure'].value_counts()

# Separate class
class_0 = data_csv[data_csv['failure'] == 0]
class_1 = data_csv[data_csv['failure'] == 1]# print the shape of the class
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)

# class_0_under = class_0.sample(class_count_1)

# test_under = pd.concat([class_0_under, class_1], axis=0)

# print("total class of 1 and 0:",test_under['failure'].value_counts())# plot the count after under-sampeling
# test_under['failure'].value_counts().plot(kind='bar', title='count (failure)')

# class_1_over = class_1.sample(class_count_0, replace=True)

# test_over = pd.concat([class_1_over, class_0], axis=0)

# print("total class of 1 and 0:",test_under['Class'].value_counts())# plot the count after under-sampeling
# test_over['Class'].value_counts().plot(kind='bar', title='count (target)')
class_1_over = class_1.sample(class_count_0, replace=True)

test_over = pd.concat([class_1_over, class_0], axis=0)

print("total class of 1 and 0:",test_over['failure'].value_counts())# plot the count after under-sampeling
test_over['Class'].value_counts().plot(kind='bar', title='count (target)')

x,y= PA.pca_preprocess(test_over,['attribute1','attribute2','attribute3','attribute4','attribute5','attribute6','attribute7','attribute8'  ,'attribute9'],['failure'])
PCA_result=PA.pca_analysis(x,y,3)


#Overall variance captured 53%

#Risk Factors : Sugar, Blood Glucose, Blood Urea, Serum Creatinie, Haemoglobin, Packed Cell Volume, Sodium, Potassium

#Machine Learning Classification Model 
X_train, X_test, y_train, y_test=CLA.train_test_data(data_csv, 0, 0.2, ['attribute1','attribute2','attribute3','attribute4','attribute5','attribute6','attribute7','attribute8'  ,'attribute9'],['failure'])
model=CLA.XGboost_Model(X_train, X_test, y_train, y_test)
Result=CLA.performance_evalute(model, X_test, y_test)


