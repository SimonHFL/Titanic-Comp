import csv as csv
import numpy as np
import pylab as P
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 

df = pd.read_csv('train.csv', header=0)


# set gender to 0/1 value
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1}).astype(int)

# set missing ages
median_ages = np.zeros((2,3))

for i in range(0,2):
	for j in range(0,3):
		median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()


print(median_ages)

df['AgeFill'] = df['Age']		

for i in range(0,2):
	for j in range(0,3):
		df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

# set embarked port to 0/1/2

df['Embarked'] = df['Embarked'].map( { 'C':0, 'S':1, 'Q':2} )
# temporary fix
df['Embarked'][df['Embarked'].isnull()] = 0

#print(df.dtypes[df.dtypes.map(lambda x: x=='object')])

#remove columns
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Age'], axis = 1)

train_data = df.values










test_df = pd.read_csv('test.csv', header=0)

# set gender to 0/1 value
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1}).astype(int)

# set missing ages
median_ages = np.zeros((2,3))

for i in range(0,2):
	for j in range(0,3):
		median_ages[i,j] = test_df[(test_df['Gender'] == i) & (test_df['Pclass'] == j+1)]['Age'].dropna().median()


print(median_ages)

test_df['AgeFill'] = test_df['Age']		

for i in range(0,2):
	for j in range(0,3):
		test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender == i) & (test_df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]


#set embarked port to 0/1/2

test_df['Embarked'] = test_df['Embarked'].map( { 'C':0, 'S':1, 'Q':2} )
# temporary fix
test_df['Embarked'][test_df['Embarked'].isnull()] = 0

# correct fare
median_fare = np.zeros(3)
for f in range(0,3):                                              # loop 0 to 2
    median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
for f in range(0,3):                                              # loop 0 to 2
    test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]



ids = test_df['PassengerId'].values


test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Age'], axis = 1)

test_data = test_df.values



print 'Training...'

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])


output = forest.predict(test_data).astype(int)


predictions_file = open('myfirstforest.csv', 'wb')
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['PassengerId', 'Survived'])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print('Done')