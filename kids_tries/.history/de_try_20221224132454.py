SEED = 1412
Test_Ratio = 0.2

import os 
import json 
import pandas as pd

Home = os.getcwd()
DataFolder = 'Data'
DataFolder = os.path.join(Home,DataFolder)

MainResultFolder = 'Main_Result'
try:
    os.mkdir(MainResultFolder)
except:
    print('Already Got this Data Folder too')
MainResultFolder = os.path.join(Home,MainResultFolder)

Supplementary_result = 'Supplementary_Result'
try:
    os.mkdir(Supplementary_result)
except:
    print('High on Energy')
Supplementary_result = os.path.join(Home,Supplementary_result)

Train_Folder = os.path.join(DataFolder,'Train.csv')
Test_Folder = os.path.join(DataFolder,'Test.csv')

Train_File = os.path.join(DataFolder,'Train.csv')
Test_File = os.path.join(DataFolder,'Test.csv')

Train = pd.read_csv(Train_File)
Test = pd.read_csv(Test_File)
col_lst = list(Test.columns)
col_lst.remove('Answers')
col_lst = [i for i in col_lst if not i.startswith('Unnamed')]
X_train = Train[col_lst]
X_test = Test[col_lst]
y_train = Train['Answers']
y_test = Test['Answers']

#print(X_train,y_train,X_test,y_test,sep='\n')

import json
import numpy as np 
import niapy
from sklearn_nature_inspired_algorithms.model_selection import NatureInspiredSearchCV
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

clf.fit(X_train, y_train)

val = clf.score(X_test, y_test)

print(val)

from niapy.algorithms.basic import GeneticAlgorithm

Algo = GeneticAlgorithm()
#Algo.set_parameters(population_size=population_size,tournament_size=tournament_size,mutation_rate=mutation_rate,crossover_rate=crossover_rate,selection=selection,crossover=crossover,mutation=mutation,seed=SEED) 

nia_mdl = NatureInspiredSearchCV(
    estimator=RandomForestClassifier(),#(n_estimators=n_estimator,criterion=criterion,max_features=max_feature),         
    param_grid={},
    algorithm=GeneticAlgorithm(),
    runs=1,
)

nia_mdl.fit(X_train, y_train)
#help(nia_mdl)
val = nia_mdl.score(X_test, y_test)

print(val)