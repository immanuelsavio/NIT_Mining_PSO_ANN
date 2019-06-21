import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
import os;
path="/home/imman/GitHub/NIT_Mining_PSO_ANN/"
os.chdir(path)
os.getcwd()
#Variables
df=pd.read_csv("data.csv")
#print(df.describe()) #to understand the dataset

y= df["PPV"]
x=df.drop("PPV",axis=1)

#print(x.head)
#print(y.head)

y_max = max(y)
#print(y_max)

#y = y/y_max
#print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2)

model = Sequential()
model.add(Dense(12, input_dim=6, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
print(x)
print(" ")
print(y)

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

history = model.fit(X_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

Xnew = np.array([[677,236,3,2.5,63,9.5]])#,4.42
Ynew = model.predict(Xnew)
print(Ynew)#*y_max)/2)