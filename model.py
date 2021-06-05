import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
dataset= pd.read_csv("diabetes.csv")
dataset.head()
dataset.shape
dataset.describe()
dataset['Outcome'].value_counts()
dataset.groupby('Outcome').mean()
x=dataset.drop(columns='Outcome',axis=1)
y=dataset['Outcome']
print(x)
print(y)
scaler=StandardScaler()
scaler.fit(x)
sd=scaler.transform(x)
print(sd)
x=sd
y=dataset['Outcome']
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
print(x.shape,x_train.shape,x_test.shape)
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)
x_train_prediction=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy score of the training data :',training_data_accuracy)
x_test_prediction=classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('Accuracy score of the test data :',test_data_accuracy)
input_data=(10,139,80,0,0,27.1,1.441,57)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
std_data=scaler.transform(input_data_reshaped)
print(std_data)
prediction= classifier.predict(std_data)
print(prediction)
import pickle
with open('model_pickle','wb')as f:
    pickle.dump(classifier,f)
with open('model_pickle','rb')as f:
    mp=pickle.load(f)