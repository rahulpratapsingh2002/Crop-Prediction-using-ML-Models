# Decision tree regressor
# importing the required libraries
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


city = input("Enter the city name: ")
# Reading the csv file
data = pd.read_csv('cpdata.csv')


# Creating dummy variable for target i.e label
label = pd.get_dummies(data.label).iloc[: , 1:]
data = pd.concat([data,label],axis=1)
data.drop('label', axis=1,inplace=True)

train = data.iloc[:, 0:4].values
test = data.iloc[:, 4:].values

# Dividing the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.3)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing Decision Tree classifier

clf = DecisionTreeRegressor()

# Fitting the classifier into training set
clf.fit(X_train,y_train)
pred = clf.predict(X_test)


# Finding the accuracy of the model
a = accuracy_score(y_test, pred)
print("The accuracy of this model is: ", a*100)
print('the mean absolute training  error is : ', mean_absolute_error(y_test, pred))

# Getting data
url = 'http://api.openweathermap.org/data/2.5/weather?q='+city+'&appid=d7ffffa8d2932fb3242ce187acd3df0a&units=metric'
req = requests.get(url)
data = req.json()
pprint(data)

temp = data['main']['temp']
humid = data['main']['humidity']


