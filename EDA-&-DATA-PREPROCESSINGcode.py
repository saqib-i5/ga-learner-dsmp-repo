# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(path)
#Code starts here

# data['Rating'].hist()
x = data['Rating'] <= 5
data = data[x]

data.hist()
#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()
percent_null = (total_null/data.isnull().count())

missing_data = pd.concat([total_null, percent_null], axis=1 , keys=['Total','Percent'])
print(missing_data)

data = data.dropna()

total_null_1 = data.isnull().sum()
percent_null_1 = (total_null/data.isnull().count())
missing_data_1 = pd.concat([total_null_1, percent_null_1], axis=1 , keys=['Total','Percent'])
print(missing_data_1)
# code ends here



# --------------

#Code starts here
g = sns.catplot(x="Category",y="Rating",data=data, kind="box", height = 10)
g.set_titles("Rating vs Category [BoxPlot]")


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
# data['Installs'].value_counts()
data['Installs'] = data['Installs'].str.replace('+', '')
data['Installs'] = data['Installs'].str.replace(',', '')
# data['Installs'] = [x.strip(',') for x in data['Installs']]
# data['Installs'] = [x.strip('+') for x in data['Installs']]
data['Installs'] = pd.to_numeric(data['Installs'], downcast='integer')

le = LabelEncoder()
data['Installs'] = le.fit_transform(data['Installs'])
data['Installs'] = le.fit_transform(data['Installs'])

g = sns.regplot(x="Installs", y="Rating" , data=data)
# g.set_titles('Rating vs Installs [RegPlot]')


#Code ends here



# --------------
#Code starts here
data['Price'].value_counts()
data['Price'] = data['Price'].str.replace('$', '')
data['Price'] = pd.to_numeric(data['Price'])

data['Price'] = pd.to_numeric(data['Price'], downcast='float')

g = sns.regplot(x="Price", y="Rating" , data=data)
# g.set_titles('Rating vs Price [RegPlot]')

#Code ends here


# --------------

#Code starts here
# data['Genres'].unique()
data['Genres'] = data.Genres.str.split(';').str[0]
# data['Genres'] = pd.to_numeric(data['Genres'], downcast='integer')
# data['Genres'] = data['Genres'].astype('category')
# data['Genres'] = data['Genres'].cat.codes
# print(data['Genres'])

gr_mean = data[['Genres','Rating']].groupby('Genres' ,as_index=False).mean().sort_values(by='Rating')


# gr_mean.describe()
# gr_mean =  gr_mean.sort_values(by = 'Rating')
print(gr_mean.iloc[0], gr_mean.iloc[-1])
#Code ends here


# --------------

#Code starts here
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()

data['Last Updated Days'] = max_date - data['Last Updated']
data['Last Updated Days'] = data['Last Updated Days'].dt.days
sns.regplot(x="Last Updated Days",y="Rating",data=data)
plt.title("Rating vs Category [BoxPlot]")
plt.show()



#Code ends here


