
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df=pd.read_csv("inflation.csv")

cdf=df[['Year','Jan','Feb']]
viz=cdf[['Year','Jan','Feb']]
viz.hist()

#plt.show()

plt.scatter(cdf.Year, cdf.Jan, color='red')
plt.xlabel("year")
plt.ylabel("Jan")
plt.show()


plt.scatter(cdf.Year, cdf.Feb, color='green')

plt.xlabel("yaer")
plt.ylabel("Feb")
plt.show()

#Для обучения машины нужно разделить датасет разделить на 20/80, где 80-тест-дата(тренировачный датасет)

msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]

from sklearn import linear_model

regr=linear_model.LinearRegression()

train_x=np.asanyarray(train[['Year']])
train_y=np.asanyarray(train[['Jan']])
regr.fit(train_x, train_y)

print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)


plt.scatter(train.Year, train.Jan, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0],'-r')

plt.xlabel("year")
plt.ylabel("Jan")
plt.show()

from sklearn.metrics import r2_score

test_x=np.asanyarray(test[['Year']])
test_y=np.asanyarray(test[['Jan']])
test_y_=regr.predict(test_x)
plt.scatter(test_x, test_y, color='yellow')
plt.plot(test_x, regr.coef_[0][0]*test_x + regr.intercept_[0],'-r')
plt.xlabel("year")
plt.ylabel("Jan")
plt.show()

print("Средняя абсолютная ошибка: %.2f" % np.mean(np.absolute(test_y_-test_y)))
#print("Остаточная сумма равенств: %.2f" % np.mean((test_y_-test_y)**2))
print("Коэффициент детерминации(R2 score): %.2f" % r2_score(test_y_, test_y))
















































































































































