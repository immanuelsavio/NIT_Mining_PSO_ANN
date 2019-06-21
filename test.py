from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
from utils import normalize
from sklearn.model_selection import train_test_split
 

df = pd.read_csv("data.csv",sep=',')
#print(df.head)
x = normalize(df)
X = pd.DataFrame(x).drop(labels="PPV", axis=1)
y = pd.DataFrame(df.PPV)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state = 1)
# create a Linear Regressor   


lin_regressor = LinearRegression()

# pass the order of your polynomial here  
poly = PolynomialFeatures(6)

# convert to be used further to linear regression
X_transform = poly.fit_transform(X_train)

# fit this to Linear Regressor
lin_regressor.fit(X_transform,y_train) 

# get the predictions
y_preds = lin_regressor.predict(X_test)
