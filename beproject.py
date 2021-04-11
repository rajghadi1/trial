import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
df=pd.read_csv(r"C:\Users\Raj Ghadi\PycharmProjects\BE_PROJECT\parkinsons.data")

x=df.loc[:,df.columns != 'status'].values[:,1:]
y=df.loc[:,'status'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
lr=LogisticRegression()
lr.fit(x_train,y_train)



y_pred1=lr.predict(x_test)
print("Accuracy Score is",accuracy_score(y_test,y_pred1)*100,"%" )
print(lr.score(x_test,y_test))
print(lr.score(x_train,y_train))

print(lr.predict([[119.99200,157.30200,74.99700,0.00784,0.00007,0.00370,0.00554,0.01109,0.04374,0.42600,0.02182,0.03130,0.02971,0.06545,0.02211,21.03300,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654]]))

from pickle4 import pickle
pickle.dump(lr,open("beproject.pkl",'wb'))
model=pickle.load(open("beproject.pkl",'rb'))
print(model.predict([[119.99200,157.30200,74.99700,0.00784,0.00007,0.00370,0.00554,0.01109,0.04374,0.42600,0.02182,0.03130,0.02971,0.06545,0.02211,21.03300,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654]]))