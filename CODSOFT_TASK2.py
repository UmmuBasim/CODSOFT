*   Author : Thayyiba Thaha
*   Title  : Iris Flower Classification

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import DecisionTreeClassifier from sklearn.tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv(r"C:\Users\Junai\Downloads\IRIS (1).csv")

data

data.describe()

data.shape

data.info()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(data['species'])

data['species'] = le.transform(data['species'])

visualisation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(data['sepal_length'],data['sepal_width'],data['petal_width'],c=data['species'])
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_width')
plt.title('3D SCATTERPLOT')
plt.show()



fig=plt.figure(figsize=(10,7))
ax=fig.add_subplot(111,projection='3d')
ax.scatter(data['sepal_length'],data['sepal_width'],data['petal_length'],c=data['species'])
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_length')
plt.title('3D SCATTERPLOT')
plt.show()


x=data.drop("species",axis=1)
y=data["species"]

y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=455)

dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(accuracy_score(y_test,y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred1=lr.predict(x_test)
print(accuracy_score(y_test,y_pred1))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred1)

