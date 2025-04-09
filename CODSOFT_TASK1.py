*   Author : Thayyiba Thaha
*   Title  : Titanic Survival Prediction


#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load dataset
data = pd.read_csv(r"C:\Users\Junai\Downloads\Titanic-Dataset.csv")
data.head()

data.info()

data.shape

data.describe()

data["Survived"].value_counts()

sns.countplot(x=data['Survived'], hue=data["Sex"])

sns.countplot(x=data['Survived'], hue=data["Pclass"])

# Handle missing values
data["Age"].fillna(data["Age"].mean(), inplace=True)


data.info()

data.dropna(subset=['Embarked'], inplace=True)

data.info()

label_encod = LabelEncoder()
for col in ['Sex', 'Embarked']:
    if col in data.columns:
        data[col] = label_encod.fit_transform(data[col])

data.head()

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'
X = data[features]
y = data[target]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


