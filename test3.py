import csv as csv
import numpy as np
import pylab as P
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 


def setGender(df):
	df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1}).astype(int)
	return df

def setEmbarked(df):
	df['Embarked'] = df['Embarked'].map( { 'C':0, 'S':1, 'Q':2} )
	# temporary fix
	df['Embarked'][df['Embarked'].isnull()] = 0

	return df

def setAgeFill(df):

	if len(df.Age[df.Age.isnull() ]) > 0:

		median_ages = np.zeros((2,3))

		for i in range(0,2):
			for j in range(0,3):
				median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

		df['AgeFill'] = df['Age']		

		for i in range(0,2):
			for j in range(0,3):
				df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

	return df

def dropColumns(df):
	df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId', 'Age'], axis = 1)
	return df

def setFare(df):

	if len(df.Fare[ df.Fare.isnull() ]) > 0:
	    
	    median_fare = np.zeros(3)

	    for f in range(0,3):                                              # loop 0 to 2
	        median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()

	    for f in range(0,3):                                              # loop 0 to 2
	        df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]

	return df





df = pd.read_csv('train.csv', header=0)

df = setGender(df)

df = setEmbarked(df)

df = setAgeFill(df)

df = dropColumns(df)

df = setFare(df)

train_data = df.values







test_df = pd.read_csv('test.csv', header=0)

test_df = setGender(test_df)

test_df = setAgeFill(test_df)

test_df = setEmbarked(test_df)

test_df = setFare(test_df)

ids = test_df['PassengerId'].values

test_df = dropColumns(test_df)

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