#importing all the required packages
import pandas as pd
import numpy as np
import random as rnd
import pydot as pydt
import pydotplus as pydtp
import StringIO


# importing the packages needed for visualization
import seaborn as sbs
import matplotlib.pyplot as plt

# importing the needed packages from scikit
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from PIL import Image

#loading the Training and Testing Data and Combining them to a single set for few feature scalings
Training_Set = pd.read_csv('C:/Users/Kishan Kandirelli/Desktop/Titanic/train.csv')
Testing_Set = pd.read_csv('C:/Users/Kishan Kandirelli/Desktop/Titanic/test.csv')
combine = [Training_Set, Testing_Set]

#Pringting the columns in Training Set
print(Training_Set.columns.values)
#printing the first few rows from the training set
print Training_Set.head()
#printing the last few rows from the test set
print Training_Set.tail()

#printing the data types used in the training set
print Training_Set.info()
print('##'*40)
#printing the data types used in the testing set
print Testing_Set.info()

#To describe all the input dataframes
print Training_Set.describe()
#To limit the result to Object dtypes, finding the distribution of the categorical features "O" represents all categorical or Object datatypes
print Training_Set.describe(include=['O'])

#pivoting features against each other, using only categorical features from the datasets.

#Pivoting Pclass, Survived and grouping them based on pclass feature and their mean scores for survived feature
print "Pivoting Pclass & Survived:"
print Training_Set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print "\n"
#Pivoting Sex, Survived and grouping them based on Sex feature and their mean scores for survived feature
print "Pivoting Sex & Survived:"
print Training_Set[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print "\n"
#Pivoting Pclass, SibSp and grouping them based on SibSp feature and their mean scores for survived feature
print "Pivoting Siblings/Spouses & Survived:"
print Training_Set[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print "\n"
#Pivoting Parch, Survived and grouping them based on Parch feature and their mean scores for survived feature
print "Pivoting Parent/Child & Survived:"
print Training_Set[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print "\n"


#Using FacetGrid to plot a histogram for Age and survived
g = sbs.FacetGrid(Training_Set, col='Survived')
g.map(plt.hist, 'Age', bins=20)
#plt.show()

#Using facetgrid method to plot a histogram for survived as column and pclass as row
grid = sbs.FacetGrid(Training_Set, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
#plt.show()

#Using Facetgrid checking if Embarked feature has made any contribution with Pclass and male and female.
grid = sbs.FacetGrid(Training_Set, row='Embarked', size=2.2, aspect=1.6)
grid.map(sbs.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', hue_order=["female", "male"], order=None)
grid.add_legend()
#plt.show()

#plotting a barplot to see male and female at a specific embarkment feature contribution
grid = sbs.FacetGrid(Training_Set, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sbs.barplot, 'Sex', 'Fare', alpha=.5, ci=None, order=["female", "male"])
grid.add_legend()
#plt.show()


print "########################## STARTING the PRE-PROCESSING of the DATASETS ############################## "

#printing the dataset information like number of tuples and attributes in both the training and test dataset
print("Before", Training_Set.shape, Testing_Set.shape, combine[0].shape, combine[1].shape)

#Dropping Ticket and Cabin feature from both Training and Test set as there is not contribution made. and most of them are alphaneumerical and cannot be used in any manner in our dataaset
Training_Set = Training_Set.drop(['Ticket', 'Cabin'], axis=1)
Testing_Set = Testing_Set.drop(['Ticket', 'Cabin'], axis=1)
combine = [Training_Set, Testing_Set]

#printing the dataset information like number of tuples and attributes in both the training and test dataset
print("After", Training_Set.shape, Testing_Set.shape, combine[0].shape, combine[1].shape)

#Extracting different titles used in the Name Feature, to check for their contribution.
for dset in combine:
    dset['Title'] = dset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


#Printing the different titles used across the Data set using a cross table format, which gives us the rarely used titles list included
print('_'*40)
print pd.crosstab(Training_Set['Title'], Training_Set['Sex'])

#Classifying the few rarely used titles as Rare in both the training and test set
print "Classifying the few rarely used titles as Rare in both the training and test set"
for dset in combine:
    dset['Title'] = dset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dset['Title'] = dset['Title'].replace('Mlle', 'Miss')
    dset['Title'] = dset['Title'].replace('Ms', 'Miss')
    dset['Title'] = dset['Title'].replace('Mme', 'Mrs')

#printing Titles and their survived score mean
print "printing Titles and their survived score mean"
print Training_Set[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#Converting the categorical titles to ordinal values  1, 2, 3, 4, 5
print "Converting the categorical titles to ordinal values  1, 2, 3, 4, 5"
convert_title = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dset in combine:
    dset['Title'] = dset['Title'].map(convert_title)
    dset['Title'] = dset['Title'].fillna(0)

#printing the first few tuples from the training dataset
print Training_Set.head()

#Dropping the Name, PassengerID feature from the Training Dataset as they do no Contribution to Survived feature
Training_Set = Training_Set.drop(['Name', 'PassengerId'], axis=1)
#Dropping the Name feature from the Test Dataset as it is not needed
Testing_Set = Testing_Set.drop(['Name'], axis=1)

#combining the Training and Test datasets and printing them
combine = [Training_Set, Testing_Set]
print Training_Set.shape, Testing_Set.shape

#converting the Sex categorical feature to ordinal values 0, 1, i.e. Female=1, Male=0
print "converting the Sex categorical feature to ordinal values 0, 1, i.e. Female=1, Male=0"
for dset in combine:
    dset['Sex'] = dset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

print Training_Set.head()

#Using facetgrid method to plot a histogram for Ordinal values of Sex as column and pclass as row
grid = sbs.FacetGrid(Training_Set, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
print "\n"

#creating an 2X3 array of zeros for 6 entries.
guess_ages = np.zeros((2,3))
print guess_ages
print "\n"
#plt.show()

#for ordinal Sex value 0 ,1 and pclass values 1, 2, 3. we calculate age values from 6 different combinations
#and use the random of all the medians to fill in the missing age values in the dataset
for dset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dset[(dset['Sex'] == i) & (dset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dset.loc[ (dset.Age.isnull()) & (dset.Sex == i) & (dset.Pclass == j+1), 'Age'] = guess_ages[i,j]

    dset['Age'] = dset['Age'].astype(int)

print Training_Set.head()
print "\n"
#printing the the 6 randomly calculated age medians from the dataset
print "These are the 6 randomly calculated age medians to be used to fill the Missing Age values: "
print guess_ages
print "\n"

#Creating a new feature called "Age Band" of scale 5, for simplifying our task
Training_Set['AgeBand'] = pd.cut(Training_Set['Age'], 5)
print Training_Set[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

print "\n"
for dset in combine:    
    dset.loc[ dset['Age'] <= 16, 'Age'] = 0
    dset.loc[(dset['Age'] > 16) & (dset['Age'] <= 32), 'Age'] = 1
    dset.loc[(dset['Age'] > 32) & (dset['Age'] <= 48), 'Age'] = 2
    dset.loc[(dset['Age'] > 48) & (dset['Age'] <= 64), 'Age'] = 3
    dset.loc[ dset['Age'] > 64, 'Age']
print Training_Set.head()
print "\n"


## removing AgeBand Feature as we have already changed the required Age feature to ordinals as 0, 1, 2, 3 based on AgeBands.

Training_Set = Training_Set.drop(['AgeBand'], axis=1)
combine = [Training_Set, Testing_Set]
print Training_Set.head()
print "\n"

##Creating new feature Familysize using "Parch" and "SibSp" so we can drop Parch, SipSp features later
for dset in combine:
    dset['FamilySize'] = dset['SibSp'] + dset['Parch'] + 1

Training_Set[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print Training_Set[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

##Creating a new feature called isAlone, for the tuple with familysize 1
for dset in combine:
    dset['IsAlone'] = 0
    dset.loc[dset['FamilySize'] == 1, 'IsAlone'] = 1

Training_Set[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print Training_Set.head()

Training_Set = Training_Set.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
Testing_Set = Testing_Set.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [Training_Set, Testing_Set]
print "\n"
print Training_Set.head()
print "\n"


#Creating a new feature Age*Class using Age and PClass but multiplying Age(0,1,2,3)*PClass(1,2,3)
for dset in combine:
    dset['Age*Class'] = dset.Age * dset.Pclass

Training_Set.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
print Training_Set.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
print Training_Set.head()
print "\n"

##finding the most frequently used port from the data set to fill Embarked field
freq_port = Training_Set.Embarked.dropna().mode()[0]
print "Frequently used port in dataset:", freq_port
print "\n"
for dset in combine:
    dset['Embarked'] = dset['Embarked'].fillna(freq_port)
    
Training_Set[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print Training_Set[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print "\n"
#Changing the Emabrked field to neumerical value

for dset in combine:
    dset['Embarked'] = dset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
print "\n"
print Training_Set.head()
print "\n"


Testing_Set['Fare'].fillna(Testing_Set['Fare'].dropna().median(), inplace=True)
print Testing_Set.head()
print "\n"

#creating new feature FareBand
Training_Set['FareBand'] = pd.qcut(Training_Set['Fare'], 4)
Training_Set[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
print "\n"
print Training_Set[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
print "\n"
#changing the fareBand values to ordinal values
for dset in combine:
    dset.loc[ dset['Fare'] <= 7.91, 'Fare'] = 0
    dset.loc[(dset['Fare'] > 7.91) & (dset['Fare'] <= 14.454), 'Fare'] = 1
    dset.loc[(dset['Fare'] > 14.454) & (dset['Fare'] <= 31), 'Fare']   = 2
    dset.loc[ dset['Fare'] > 31, 'Fare'] = 3
    dset['Fare'] = dset['Fare'].astype(int)
##Dropping the FareBand feature as it is no longer needed.
Training_Set = Training_Set.drop(['FareBand'], axis=1)
combine = [Training_Set, Testing_Set]
print "\n"
print"this is Training data", ('#'*20) 
print Training_Set.head(10)
print "\n"


print "this is Test data", ('#'*20) 
print Testing_Set.head(10)
print "\n"

print "##################### END of PRE-PROCESSING the DATASET ################################"



################ Using Models to predict ##############
print "\n \n"
print "####### MODEL PREDICTION using CLASSFICATION and REGRESSION Techniques ###########"

#Dropping the Survived feature from the training set
trainingX = Training_Set.drop("Survived", axis=1)
#Taking the Survived feature to trainingY
trainingY = Training_Set["Survived"]
#copying only the PassengerID to testingX from the test dataset and ignoring rest of the features 
testingX  = Testing_Set.drop("PassengerId", axis=1).copy()
print trainingX.shape, trainingY.shape, testingX.shape

# Using Logistic Regression

logreg = LogisticRegression()
logreg.fit(trainingX, trainingY)
PredictY = logreg.predict(testingX)
acc_log = round(logreg.score(trainingX, trainingY) * 100, 2)
print"Logistic Regression Accuracy:", acc_log
print "\n"

#Using Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(trainingX, trainingY)
PredictY = gaussian.predict(testingX)
acc_gaussian = round(gaussian.score(trainingX, trainingY) * 100, 2)
print"Naive Bayes Accuracy:", acc_gaussian
print "\n"

print "trainingX"
print trainingX.columns.values



#Using Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(trainingX, trainingY)
PredictY = decision_tree.predict(testingX)
acc_decision_tree = round(decision_tree.score(trainingX, trainingY) * 100, 2)
print "Decision Tree Accuracy:", acc_decision_tree
print "\n"
feature_name = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "IsAlone", "Age*Class"]

d_t = tree.DecisionTreeClassifier(
    random_state = 1,
    max_depth = 5,
    min_samples_split = 2
    )
d_t_ = d_t.fit(trainingX, trainingY)
print"Decision Tree Score:", decision_tree.score(trainingX, trainingY)



tree.export_graphviz(d_t_, feature_names=feature_name, out_file="C:/Users/Kishan Kandirelli/Desktop/Titanic/Tree.dot", filled=True, rounded=True)


## Evaluating

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'Decision Tree'],
    'Score': [acc_log, acc_gaussian, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
print models.sort_values(by='Score', ascending=False)
print "\n"

Results = pd.DataFrame({
        "PassengerId": Testing_Set["PassengerId"],
        "Survived": PredictY
    })
#print submission
##final predicted output is sent to a csv file
Results.to_csv('C:/Users/Kishan Kandirelli/Desktop/Titanic/TestResults.csv', index=False)

#displaying all the graphs plotted so far
plt.show()
