# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
cols = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']
df[cols] = df[cols].replace({'\$': '', ',': ''}, regex=True)

print(df.head())
X = df.drop('CLAIM_FLAG', axis=1)
y = df['CLAIM_FLAG']
count = y.value_counts()
print(count)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=6)


# --------------
# Code starts here
X_train[cols] = X_train[cols].astype(float)
X_test[cols] = X_test[cols].astype(float)
print(X_train.isnull().sum())
print(X_test.isnull().sum())



# --------------
# Code starts here
X_train.dropna(subset=['YOJ','OCCUPATION'],inplace=True)
X_test.dropna(subset=['YOJ','OCCUPATION'],inplace=True)
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]
col = ['AGE','CAR_AGE','INCOME','HOME_VAL']

X_train[col].replace(np.nan,X_train.mean(),inplace=True)
X_test[col].replace(np.nan,X_train.mean(),inplace=True)


# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
le = LabelEncoder()
for i in columns:
    X_train[i] = le.fit_transform(X_train[i])
    X_test[i] = le.transform(X_test[i])

print(X_train.head())

# Code ends here


# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 

# Instantiate logistic regression
model = LogisticRegression(random_state = 6)

# fit the model
model.fit(X_train,y_train)

# predict the result
y_pred =model.predict(X_test)

# calculate the f1 score
score = accuracy_score(y_test, y_pred)
print(score)


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here

# Instantiate SMOTE 
smote = SMOTE(random_state=9)

# fit smote on training set
X_train, y_train = smote.fit_sample(X_train, y_train)

# code ends here

# Instantiate a standardScaler
scaler = StandardScaler()

# Fit on training set only.
X_train = scaler.fit_transform(X_train)

# Apply transform to the test set.
X_test = scaler.transform(X_test)


# --------------
# Instantiate logistic regression
model = LogisticRegression()

# fit the model
model.fit(X_train,y_train)

# predict the result
y_pred =model.predict(X_test)

# calculate the f1 score
score = accuracy_score(y_test, y_pred)
print(score)


