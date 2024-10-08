import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

trainingdata = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
trainingdata= trainingdata.set_index('id')
#trainingdata['DateTime'] = pd.to_datetime(trainingdata['DateTime'])
#trainingdata['Day'] = trainingdata['DateTime'].dt.day
#trainingdata['Hour'] = trainingdata['DateTime'].dt.hour
trainingdata = trainingdata.drop('DateTime', axis=1) # Day/Hour might be useful, but need to extract from DateTime column, dropping for now


preddata = pd.read_csv("https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/assignment3test.csv")
preddata = preddata.set_index('id')
preddata = preddata.drop('DateTime', axis = 1)
preddata = preddata.drop('meal', axis=1)

y = trainingdata['meal']
x = trainingdata.drop('meal', axis = 1)

model = RandomForestClassifier(n_estimators=1000, n_jobs = -1, random_state=42)

modelFit = model.fit(x, y)

pred = modelFit.predict(preddata)
