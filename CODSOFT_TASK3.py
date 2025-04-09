*   Author : Thayyiba Thaha
*   Title  : Sales Prediction Using Python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data=pd.read_csv(r"C:\Users\Junai\Downloads\advertising (2).csv")

data

data.info()

data.describe()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data[["TV", "Radio", "Newspaper", "Sales"]] = scaler.fit_transform(data[["TV", "Radio", "Newspaper", "Sales"]])

data.describe()

# Let's see how Sales are related with other variables using scatter plot.
sns.pairplot(data, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()


# Let's see the correlation between different variables.
sns.heatmap(data.corr(), cmap="YlGnBu", annot = True)
plt.show()

# we select TV as a relevant feature as correlation between TV and sles is 0.9
x=data["TV"]
y=data["Sales"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

lr=LinearRegression()
lr.fit(x_train.values.reshape(-1,1),y_train)

y_pred=lr.predict(x_test.values.reshape(-1,1))
r2_score(y_test,y_pred)

y_test

y_pred

coef=lr.coef_
intr=lr.intercept_

plt.scatter(y_test,y_pred)
plt.plot(y_test,intr+coef*y_test,color="red")
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()

