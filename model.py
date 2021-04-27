import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


result = pandas.read_csv(r"vaccine.csv")
model = LinearRegression()
y = result['total_vaccinated']
x = result['Days']
x = np.array(x)
x=x.reshape(-1,1)
y = np.array(y)
y=y.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model=model.fit(x_train, y_train)
yhat = model.predict(x_test)
from sklearn.metrics import r2_score
acc=r2_score(y_test, yhat)
print("Accuracy of Vaccination Prediction using Linear Regression is : {}%".format(round(acc,4)*100))
day=100
ans=0
while(ans<1500000000):
    day=day+100;
    ans = model.predict([[day]])
print("For the vaccination of the entire country(INDIA) :")
print("{} years".format(round(day/365,2)))
print(pickle.dump(model, open('model2.pkl','wb')))



dataset = pandas.read_csv(r"vaccine.csv")
c=dataset.corr()
#print(c)
# Using Lasso we got to know both days and deadth_per_day are relevant feature
y = dataset['total_vaccinated']
x = dataset[['Days','death_per_day']]
x = np.array(x)
x=x.reshape(-1,2)
y = np.array(y)
y=y.reshape(-1,1)
s=SelectFromModel(Lasso(alpha=0.01))
s=s.fit(x,y)
s.get_support()
model=LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model=model.fit(x_train, y_train)
yhat = model.predict(x_test)
acc=r2_score(y_test, yhat)
print("Accuracy of Vaccination Prediction using Lasso Regression is : {}%".format(round(acc,4)*100))
day=100
deadth=2000
ans=0
while(ans<1500000000):
    day=day+100;
    deadth=deadth+1000;
    ans = model.predict([[day,deadth]])
print("For the vaccination of the entire country(INDIA) :")
print("{} years".format(round(day/365,2)))
print(deadth)
a=model.predict([[100,2000]])
print(float(a))