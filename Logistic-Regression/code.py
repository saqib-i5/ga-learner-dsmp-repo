# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df = pd.read_csv(path)
X = df.drop('insuranceclaim', axis = 1)
y = df['insuranceclaim']

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state = 6)


# Code ends here


# --------------
import matplotlib.pyplot as plt


# Code starts here
plt.boxplot(X_train['bmi'])

q_value = X_train['bmi'].quantile(q = 0.95)
print(y_train.value_counts())

# Code ends here


# --------------
# Code starts here
relation = X_train.corr()
sns.pairplot(relation)


# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# Code starts here
# df = df.iloc[:, [3,1,5,4]]
cols = ['children','sex','region','smoker']
print(cols)


# cols = df.iloc[:, [3,1,5,4]]

fig, axes = plt.subplots(nrows = 2 , ncols = 2)
for i in range(0,2):
    for j in range(0,2):
                col = cols[i*2 + j]
                sns.countplot(x=X_train[col], hue=y_train, ax=axes[i,j])

     


            

# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here
lr = LogisticRegression(random_state=9)
grid = GridSearchCV(lr, param_grid=parameters)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

grid.fit(X_train,y_train)
g_pred = grid.predict(X_test)

accuracy = accuracy_score(y_test,g_pred)
print(accuracy)


# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics

# Code starts here
score = roc_auc_score(y_test, g_pred)
y_pred_proba = grid.predict_proba(X_test)[:,1]

fpr, tpr,_ = metrics.roc_curve(y_test, y_pred_proba)
roc_auc =roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="Logistic model, auc="+str(roc_auc))
# Code ends here


