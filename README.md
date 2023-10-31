# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Yazhini G
Register Number:212222220060  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred

Y_test

plt.scatter(X_train,Y_train,color='purple')
plt.plot(X_train,regressor.predict(X_train),color='gold')
plt.title('Hours vs Scores(Training Set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='pink')
plt.title('Hours vs Scores(Test Set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
## Dataset:
![278995575-c7816d33-6dab-45e2-8d19-9a11e9583cb5](https://github.com/Yazhini-G/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120244201/1d97ef85-709f-4981-8c69-f9fe4d25603d)

## Head values:
![278996479-7f3d7783-4601-4e70-989f-2ccbf87d0765](https://github.com/Yazhini-G/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120244201/80e00d51-d711-42f8-b4a4-4ab2fbc8572d)

## Tail values:
![278996533-5343e114-fe3a-4ad7-8058-6b81db462fdc](https://github.com/Yazhini-G/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120244201/e103e6b7-44c1-456d-b282-50a923b4b389)

## X and Y values:
![278996577-f84947e0-99a3-444c-8286-c59cc0660a4e](https://github.com/Yazhini-G/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120244201/ff7a3d37-0a2a-4ff5-b7e4-c63cf147f682)

## Predication values of X and Y:
![278996605-6ea46100-8530-4491-821e-079308a1eef5](https://github.com/Yazhini-G/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120244201/ac89d678-c5f5-4867-974b-774d01030ed7)

## MSE,MAE and RMSE:
![278996622-0f3750f1-fec0-4008-abcf-7e7b971d82a9](https://github.com/Yazhini-G/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120244201/ac1bbccf-ef3e-433f-94c0-b235cdd7f330)

## Training Set:
![278996909-088c3714-a70d-4ef0-b952-1d26c48e1fa8](https://github.com/Yazhini-G/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120244201/987f8ec6-0f82-4f3a-80be-2b3f4d8cc75a)

## Testing Set:
![278996855-aa18e6a5-11f7-410e-bbd6-89c052ff52a6](https://github.com/Yazhini-G/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/120244201/a189b100-5390-42f6-924c-f6d0350a8fa6)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
