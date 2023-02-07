import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/IRIS.csv')
df
x = df.iloc[:,0:4].values
y = df.iloc[:,4].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred #Predict_the_Output
y_test #Actual_Output
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)*100
