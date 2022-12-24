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

"""BASIC

DB
"""

from sklearn.model_selection import train_test_split

DataBase = [os.path.join(DataFolder,i) for i in os.listdir(DataFolder) if i != 'Train.csv' and i != 'Test.csv'][0]

df = pd.read_csv(DataBase)

df_col = list(df.columns)
#print(df_col)

df = df.dropna()
df = df.drop(columns=[i for i in df_col if i.startswith('Unnamed')])

df_col = list(df.columns)
#print(df_col)

y_col = 'class'
x_col = [i for i in df_col if i != y_col]

y = df[y_col]
X = df[x_col]

#Encoding Y 
set_y = list(set(y))
dict_y = dict()
for i in range(len(set_y)):
    dict_y[set_y[i]] = i
y = [dict_y[i] for i in df[y_col]]

with open(os.path.join(Supplementary_result,'Y_encoding.json'),'w') as temp_dict:
    json.dump(dict_y,fp=temp_dict)
    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = Test_Ratio, random_state = SEED)

Train = pd.DataFrame(X_train)
Train['Answers'] = y_train
Test = pd.DataFrame(X_test)
Test['Answers'] = y_test

Train_Folder = os.path.join(DataFolder,'Train.csv')
Test_Folder = os.path.join(DataFolder,'Test.csv')

Train.to_csv(Train_Folder)
Test.to_csv(Test_Folder)

"""DB

FILES
"""

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

"""FILES"""

len(Test),len(Train)

import json
import numpy as np 
import niapy
from sklearn_nature_inspired_algorithms.model_selection import NatureInspiredSearchCV
from sklearn.ensemble import RandomForestClassifier

"""
Genetic Algorithm
"""

from niapy.algorithms.basic import GeneticAlgorithm

ga_bounds = {'population_size':(10,100), 'tournament_size':(1,10), 'mutation_rate':(0.1,1.0), 'crossover_rate':(0.1,1.0),'selection':(0,2),'crossover':(0,4), 'mutation':(0,3), 'n_estimator':(10,1000),'criterion':(0,1),'max_feature':(0,1)}

def mdl_ga(population_size, tournament_size, mutation_rate, crossover_rate, selection, crossover, mutation, n_estimator, criterion, max_feature):

    population_size = int(population_size)
    tournament_size = int(tournament_size)
    n_estimator = int(n_estimator) 

    '''
    * :func:`niapy.algorithms.basic.tournament_selection`
    * :func:`niapy.algorithms.basic.roulette_selection`
    '''
    selection_name = ''
    if selection < 1:
        selection_name = 'tournament_selection'
        selection = niapy.algorithms.basic.ga.tournament_selection
    else:
        selection_name = 'roulette_selection'
        selection = niapy.algorithms.basic.ga.roulette_selection
    '''
    * :func:`niapy.algorithms.basic.uniform_crossover`
    * :func:`niapy.algorithms.basic.two_point_crossover`
    * :func:`niapy.algorithms.basic.multi_point_crossover`
    * :func:`niapy.algorithms.basic.crossover_uros`
    '''
    crossover_name = ''
    if crossover < 1:
        crossover_name = 'uniform_crossover'
        crossover =  niapy.algorithms.basic.ga.uniform_crossover
    elif crossover < 2:
        crossover_name = 'two_point_crossover'
        crossover = niapy.algorithms.basic.ga.two_point_crossover
    elif crossover < 3:
        crossover_name = 'multi_point_crossover'
        crossover = niapy.algorithms.basic.ga.multi_point_crossover
    else:
        crossover_name = 'crossover_uros'
        crossover = niapy.algorithms.basic.ga.crossover_uros
    '''
    * :func:`niapy.algorithms.basic.uniform_mutation`
    * :func:`niapy.algorithms.basic.creep_mutation`
    * :func:`niapy.algorithms.basic.mutation_uros`
    '''
    mutation_name = ''
    if mutation < 1:
        mutation_name = 'uniform_mutation'
        mutation = niapy.algorithms.basic.ga.uniform_mutation
    elif mutation < 2:
        mutation_name = 'creep_mutation'
        mutation = niapy.algorithms.basic.ga.creep_mutation
    else:
        mutation_name = 'mutation_uros'
        mutation = niapy.algorithms.basic.ga.mutation_uros


    if criterion < 0.5:
        criterion = 'gini'
    else:
        criterion = 'entropy'

    if max_feature < 0.34:
        max_feature = 'sqrt'
    elif max_feature < 0.67:
        max_feature = 'log2'
    else:
        max_feature = None
    
    Para_lst = [population_size,tournament_size,mutation_rate,crossover_rate,selection_name,crossover_name,mutation_name,n_estimator,criterion,max_feature]

    Algo = GeneticAlgorithm()
    Algo.set_parameters(population_size=population_size,tournament_size=tournament_size,mutation_rate=mutation_rate,crossover_rate=crossover_rate,selection=selection,crossover=crossover,mutation=mutation,seed=SEED) 

    nia_mdl = NatureInspiredSearchCV(
        estimator=RandomForestClassifier(n_estimators=n_estimator,criterion=criterion,max_features=max_feature),         
        param_grid={},
        algorithm=Algo,
        runs=1,
    )
    return nia_mdl, Para_lst

"""Genetic Algorithm"""



from bayes_opt import BayesianOptimization, UtilityFunction
from matplotlib import pyplot as plt 
import time

os.getcwd()

"""ENERGY FUNCTIONS"""

energy_file = [ os.path.join(os.getcwd(),x) for x in os.listdir() if x.endswith('.csv')][0]
energy_file
#always check for latest copy whenever you check

def Get_Time_Energy(Start_idx,End_idx):

    temp_df = pd.read_csv(energy_file)
    temp_df_cols = list(temp_df.columns)

    ret_dict = {'Time Taken(s)':int(End_idx-Start_idx), 'Total Power(J)':0.0, 'CPU(J)':0.0, 'Monitor(J)':0.0, 'Disk(J)':0.0, 'Base(J)':0.0}
    ret_dict_cols = list(ret_dict.keys())

    for curr_idx in range(Start_idx,End_idx):

        for i in range(1,len(ret_dict_cols)):
            ret_dict[ret_dict_cols[i]] += temp_df[temp_df_cols[i]][curr_idx]
    
    return ret_dict

"""BAYESIAN FUNCTION"""

def Optimize_and_plot(bounds,curr_mdl_fxn,Algo_name:str,Iters:int):
    os.remove(energy_file)
    time.sleep(2)
    
    print(f'Currently at {Algo_name}')

    optimizer = BayesianOptimization(f = None, pbounds = bounds, verbose = 2, random_state = SEED)
    utility = UtilityFunction(kind = "ucb", kappa = 1.96, xi = 0.01)

    #ID refers to Iteration ID, it is of format f'{Algo_name[:3]}0{Iteration_no}'
    #^ This is gonna be Primary Key

    #All Parameters
    Parameter_df = pd.DataFrame()
    col_lst = ['ID'] + list(bounds.keys())
    for col_name in col_lst:
        Parameter_df[col_name] = []

    #All results like Accuracy, time taken, Energy, CO2 
    Result_df = pd.DataFrame()
    col_lst = ['ID', 'Accuracy', 'Time Taken (s)', 'Energy Used (J)', 'Equivalent CO2 Emission (mg)']
    for col_name in col_lst:
        Result_df[col_name] = []

    #Energy Distribution
    Energy_df = pd.DataFrame()
    col_lst = ['ID', 'Time Taken(s)', 'Total Power(J)', 'CPU(J)', 'Monitor(J)', 'Disk(J)', 'Base(J)']
    for col_name in col_lst:
        Energy_df[col_name] = []

    for i in range(Iters):
        Curr_ID = f'{Algo_name[:3].upper()}-{i:>4}'.replace(' ','0')
        # Get optimizer to suggest a new parameter value to try.
        next_point = optimizer.suggest(utility)  
        # Evaluate the output of the black_box_function using the new parameter value.

        curr_mdl,Para_lst = curr_mdl_fxn(**next_point)

        Start_idx = len(pd.read_csv(energy_file))
        curr_mdl.fit(X_train,y_train)
        target = curr_mdl.score(X_test,y_test)
        End_idx = len(pd.read_csv(energy_file))
        target = target*100
        #print(Start_idx,End_idx)

        Parameter_df.loc[len(Parameter_df)] = [Curr_ID] + Para_lst

        Energy_dict = Get_Time_Energy(Start_idx,End_idx)
        #{'Time Taken(s)':int(End_idx-Start_idx+1), 'Total Power(J)':0.0, 'CPU(J)':0.0, 'Monitor(J)':0.0, 'Disk(J)':0.0, 'Base(J)':0.0}

        Result_df.loc[len(Result_df)] = [Curr_ID, target, Energy_dict['Time Taken(s)'], Energy_dict['Total Power(J)'], (17.0/72.0)*Energy_dict['Total Power(J)']]
        #['ID', 'Accuracy', 'Time Taken (s)', 'Energy Used (J)', 'Equivalent CO2 Emission (mg)']
        
        '''emission 
        1 kW-hr = 0.85 Kg of CO2 emission 
        36 e5 - 85 e4 mg
        360 - 85
        72 J - 17mg CO2 
        1 J = (17.0/72.0) mg CO2
        ''' 

        Energy_df.loc[len(Energy_df)] = [Curr_ID] + list(Energy_dict.values())
        #['ID', 'Time Taken(s)', 'Total Power(J)', 'CPU(J)', 'Monitor(J)', 'Disk(J)', 'Base(J)']

        try:
            # Update the optimizer with the evaluation results. This needs to be in try-except
            # to prevent repeat errors from occuring.
            optimizer.register(params = next_point, target = target)
        except:
            print('What was that?')
            pass

    Result_df.to_csv(os.path.join(MainResultFolder,f'{Algo_name}.csv')) 
    Parameter_df.to_csv(os.path.join(Supplementary_result,f'{Algo_name}_Parameter.csv'))
    Energy_df.to_csv(os.path.join(Supplementary_result,f'{Algo_name}_Energy_Distribution.csv'))

    plt.plot(range(1, 1+len(optimizer.space.target)), optimizer.space.target, "-o")
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.show()
    plt.savefig(os.path.join(Supplementary_result,f'{Algo_name}_Iterations.png'))
    plt.show()

"""BAYESIAN FUNCTION"""


Optimize_and_plot(bounds=ga_bounds,curr_mdl_fxn=mdl_ga,Algo_name='GA',Iters=50)
