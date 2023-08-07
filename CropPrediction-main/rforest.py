import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


data = pd.read_csv('cpdata.csv')
label = pd.get_dummies(data.label).iloc[:, 1:]
data = pd.concat([data, label], axis=1)
data.drop('label', axis=1, inplace=True)

X = data.iloc[:, 0:4]
Y = data.iloc[:, 4:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=500, random_state=0)
regressor.fit(X_train, Y_train)
pred=regressor.predict(X_test)

a=r2_score(Y_test,pred)
print(f"The accuracy of this model is: {a*100}%")

