import csv as csv
import numpy as np
import pylab as P


csv_file_object = csv.reader(open('train.csv', 'rb'))
header = csv_file_object.next()
data = []

for row in csv_file_object:
	data.append(row)
data = np.array(data)

import pandas as pd	
df = pd.read_csv('train.csv', header=0)

df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]

"""
for i in range(1,4):
	print i, len(df[(df['Sex'] == 'male') & (df['Pclass'] == i) ])
"""
"""

df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()
"""
df['Gender'] = 4
df['Gender'] = df['Sex'].map(lambda x: x[0].upper() )
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1}).astype(int)

#df['Embarked'] = df['Embarked'].map( {'S': 0, 'Q': 1, 'C':2})

#print(df['Embarked'])

median_ages = np.zeros((2,3))

for i in range(0,2):
	for j in range(0,3):
		median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()
print(median_ages)
df['AgeFill'] = df['Age']

for i in range(0,2):
	for j in range(0,3):
		df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]
#print(df.AgeFill)

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']

df['Age*Class'] = df.AgeFill * df.Pclass

#df['Age*Class'].dropna().hist()

#df['FamilySize'].dropna().hist()
#P.show()

#print(df.dtypes[df.dtypes.map(lambda x: x=='object')])

df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis = 1)


#test = df[df['Age'].isnull()][['Gender','Pclass','Age','AgeFill', 'AgeIsNull']].head(10)

train_data = df.values
#print(train_data)


#print(df.describe())
