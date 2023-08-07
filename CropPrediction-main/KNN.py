# KNN model
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv('cpdata.csv')

scaler = StandardScaler()
x = scaler.fit_transform(df.drop(['label'], axis=1))
fin_df = pd.DataFrame(x, columns=['Temprature', 'Humidity', 'PH', 'Rainfall'])
lab = df.label
values = lab.values

fin_df['Label'] = values

X_train, X_test, y_train, y_test = train_test_split(fin_df.drop('Label', axis=1), fin_df.Label, test_size=0.33)

knn = KNeighborsClassifier(algorithm='kd_tree')
knn.fit(X_train, y_train)
knn_pres = knn.predict(X_test)

actuals = y_test
a = accuracy_score(actuals, knn_pres)
print("The accuracy of this model is: ", a*100)


