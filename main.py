import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
df=pd.read_csv("data.csv")
#print(df.describe()) #to understand the dataset

y_val= df["PPV"]
x_data=df.drop("PPV",axis=1)


X_train, X_eval,y_train,y_eval=train_test_split(x_data,y_val,test_size=0.2,random_state=7)

scaler_model = MinMaxScaler()
scaler_model.fit(X_train)

X_train=pd.DataFrame(scaler_model.transform(X_train),columns=X_train.columns,index=X_train.index)

scaler_model.fit(X_eval)

X_eval=pd.DataFrame(scaler_model.transform(X_eval),columns=X_eval.columns,index=X_eval.index)

feat_cols=[]    
for cols in df.columns[:-1]:
    column=tf.feature_column.numeric_column(cols)
    feat_cols.append(column)
    
#print(feat_cols)

#The estimator model
model=tf.estimator.DNNRegressor(hidden_units=[6,10,6],feature_columns=feat_cols)

#the input function
print("Started")
input_func=tf.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=10,num_epochs=1000,shuffle=True)
model.train(input_fn=input_func,steps=1000)
train_metrics=model.evaluate(input_fn=input_func,steps=1000)
pred_input_func=tf.estimator.inputs.pandas_input_fn(x=X_eval,y=y_eval,batch_size=100,num_epochs=1,shuffle=False)
preds=model.predict(input_fn=pred_input_func)
print(" ")
print(y_eval)
predictions=list(preds)
final_pred=[]
for pred in predictions:
    final_pred.append(pred["predictions"])

#p = cross_val_score(model, X_eval, y_eval)
#print(final_pred)
print(preds)
print("Ended")