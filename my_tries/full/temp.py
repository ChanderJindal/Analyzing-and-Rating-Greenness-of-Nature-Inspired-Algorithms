Home = os.getcwd()
DataFolder = 'Data'
DataFolder = os.path.join(Home,DataFolder)

BayesianResultFolder = 'Bayesian_Result'
BayesianResultFolder = os.path.join(Home,BayesianResultFolder)

Energy_result = 'Energy_Result'
Energy_result = os.path.join(Home,Energy_result)



NIA_Name = {0:'Cuckoo Search',1:'Fire Fly',2:'Bat',3:'Self Adaptive Bat',4:'Particle Swarm', 5:'Camel Algorithm'}

NIA_pbounds_lst = [CucKoo_pbounds, FireFly_pbounds, BAT_pbounds, SABA_pbounds, PSA_pbounds, camel_bounds]

save_fxn = [Save_camel_df,Save_PSA_df,Save_SBA_df,Save_Bat_df,Save_Firefly_df,Save_Cuckoo_df]

update_fxn = [camel_value_update,PSA_value_update,SBA_value_update,Bat_value_update,Firefly_value_update,Cuckoo_value_update]
#camel_value_update(para_str,val,time_taken,carbon)
mdl_fxn = [mdl_camel, mdl_PSA, mdl_SBA,mdl_Bat,mdl_Firefly,mdl_Cuckoo]
#mdl_camel(para_str)