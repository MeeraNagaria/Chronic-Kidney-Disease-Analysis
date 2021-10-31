from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

def train_test_data(data,seed,test_size,features_name,target):
    
    X = data[features_name]
    y = data[target]
    X_trans=X.eq('yes').mul(1)
    

    X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=test_size, random_state=seed)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
    
    
def Log_Reg_Model (X_train, X_test, y_train, y_test) :
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
      .format(logreg.score(X_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}'
      .format(logreg.score(X_test, y_test)))
    return logreg
    
    
def XGboost_Model(X_train, X_test, y_train, y_test):
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    # make predictions for test data
    y_pred = xgb.predict(X_test)
    #predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return xgb

def SVM_Model(X_train, X_test, y_train, y_test):    
      svm = SVC()
      svm.fit(X_train, y_train)
      print('Accuracy of SVM classifier on training set: {:.2f}'
            .format(svm.score(X_train, y_train)))
      print('Accuracy of SVM classifier on test set: {:.2f}'
            .format(svm.score(X_test, y_test)))
      return svm
  
    
  

def Random_Forest(X_train, X_test, y_train, y_test):
        rfc = RandomForestClassifier()
        rfc.fit(X_train, y_train)
        rfc_predict = rfc.predict(X_test)# check performance
        print('ROCAUC score:',roc_auc_score(y_test, rfc_predict))
        print('Accuracy score:',accuracy_score(y_test, rfc_predict))
        print('F1 score:',f1_score(y_test, rfc_predict))    
  

def performance_evalute(model,X_test,y_test):
    
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    return print (classification_report(y_test, y_pred))
      
#Accuracy with htn, dm, pe, ane=87.5%
#htn,dm,pe = 86.25%
#htn, dm = 82.5%
