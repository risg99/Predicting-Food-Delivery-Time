import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier

train_df = pd.read_excel('Data_Train.xlsx')
test_df = pd.read_excel('Data_Test.xlsx')

train_df['Rating'] = train_df['Rating'].replace(['NEW','Opening Soon','Temporarily Closed','-'],value=0)
test_df['Rating'] = test_df['Rating'].replace(['NEW','Opening Soon','Temporarily Closed','-'],value=0)
train_df['Rating'] = pd.to_numeric(train_df['Rating'])
test_df['Rating'] = pd.to_numeric(test_df['Rating'])

train_df['Votes'] = train_df['Votes'].replace(['-'],value=0)
test_df['Votes'] = test_df['Votes'].replace(['-'],value=0)
train_df['Votes'] = pd.to_numeric(train_df['Votes'])
test_df['Votes'] = pd.to_numeric(test_df['Votes'])

train_df['Reviews'] = train_df['Reviews'].replace(['-'],value=0)
test_df['Reviews'] = test_df['Reviews'].replace(['-'],value=0)
train_df['Reviews'] = pd.to_numeric(train_df['Reviews'])
test_df['Reviews'] = pd.to_numeric(test_df['Reviews'])

cuisines = []
for i in train_df['Cuisines']:
	if i not in cuisines:
		cuisines.append(i)

locations = []
for i in train_df['Location']:
	if i not in locations:
		locations.append(i)

c,l = [],[]
for x in train_df['Location']:
	y = locations.index(x)
	l.append(y)

for x in train_df['Cuisines']:
	y = cuisines.index(x)
	c.append(y)

ct,lt = [],[]

for x in test_df['Location']:
	if x in locations:
		y = locations.index(x)
		lt.append(y)
	else:
		locations.append(x)
		y = locations.index(x)
		lt.append(y)

for x in test_df['Cuisines']:
	if x in cuisines:
		y = cuisines.index(x)
		ct.append(y)
	else:
		cuisines.append(x)
		y = cuisines.index(x)
		ct.append(y)


train_df['Minimum_Order'] = train_df['Minimum_Order'].apply(lambda x:int(x.strip('₹')))
train_df['Average_Cost'] = train_df['Average_Cost'].apply(lambda x:(x.strip('₹')))
train_df['Average_Cost'] = train_df['Average_Cost'].apply(lambda x:int(x.replace(',','')) if x != 'for' else -99)
median = int(train_df['Average_Cost'].median())
train_df["Average_Cost"] = np.where(train_df["Average_Cost"] == 0, median,train_df['Average_Cost'])

train_df['Delivery_Time'] = train_df['Delivery_Time'].apply(lambda x:int(x.strip(' minutes')))

test_df['Minimum_Order'] = test_df['Minimum_Order'].apply(lambda x:int(x.strip('₹')))
test_df['Average_Cost'] = test_df['Average_Cost'].apply(lambda x:(x.strip('₹')))
test_df['Average_Cost'] = test_df['Average_Cost'].apply(lambda x:int(x.replace(',','')) if x != 'for' else -99)
median = int(test_df['Average_Cost'].median())
test_df["Average_Cost"] = np.where(test_df["Average_Cost"] == 0, median,test_df['Average_Cost'])

knn = KNeighborsClassifier(n_neighbors = 2)
lm = linear_model.LinearRegression()
X_train = train_df.drop(['Restaurant','Delivery_Time','Location','Cuisines'],axis=1)
X_train['Location'] = l
X_train['Cuisines'] = c
X_test = test_df.drop(['Restaurant','Location','Cuisines'],axis=1)
X_test['Location'] = lt
X_test['Cuisines'] = ct
Y_train = train_df['Delivery_Time'].values
knn.fit(X_train,Y_train)
lm.fit(X_train,Y_train)
print(knn.predict(X_test)[0:5])  #[30 30 30 45 45]
print(lm.predict(X_test)[0:5])	#[39.39377194 31.04529288 36.38719694 36.96162778 35.73898515]




