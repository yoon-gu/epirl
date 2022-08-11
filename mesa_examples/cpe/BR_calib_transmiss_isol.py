#%%
from CPEmodel import CPE_Model
from CPEmodel import getNumSick, getHCWInfec
from mesa.batchrunner import BatchRunner

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
import time
import seaborn as sns
# %% STEP1,STEP2
num_iter = 50
runtime = 200 #(Days)

start_time = time.time()

# Parameters
probPatientSick = 0.01 # from data # Prob. of being hospitalizes with Infec.
probNewPatient = 0.003 # 0.053, Old Calibration # 1/2000, 2592 ticks per day
probTransmission = 0
isolationFactor = 0.75 # fix
isolationTime = 14
cleanDay = 360
isolateSick = True
ICUwashrate = 0.90
OUTSIDEwashrate = 0.90
height=11
width=32

# %% STEP3
fixed_params = {
    "prob_patient_sick" : probPatientSick, 
    "prob_new_patient" : probNewPatient, 
    # "prob_transmission" : probTransmission,
    "isolation_factor" : isolationFactor, 
    "cleaningDay" : cleanDay, 
    "isolate_sick" : isolateSick,
    "isolation_time" : isolationTime, 
    "icu_hcw_wash_rate" : ICUwashrate, 
    "outside_hcw_wash_rate" : OUTSIDEwashrate, 
    "height" : height, "width" : width 
    }

# Specify the variable I want to change separately.
beta = [0.000001, 0.000002, 0.000003, 0.000004, 0.000005, 0.000006, 0.000007, 0.000008, 0.000009]
variable_params = {"prob_transmission" : beta}
# %% STEP4
model = CPE_Model(
    prob_patient_sick=probPatientSick,prob_new_patient=probNewPatient, prob_transmission=probTransmission, 
    isolation_factor=isolationFactor,cleaningDay=cleanDay, isolate_sick=True, 
    icu_hcw_wash_rate=ICUwashrate, outside_hcw_wash_rate=OUTSIDEwashrate,
    height=height, width=width
    )

batch_run = BatchRunner(
    CPE_Model,
    variable_parameters = variable_params,
    fixed_parameters = fixed_params,
    iterations=num_iter,
    max_steps=model.ticks_in_day * runtime,
    model_reporters = {"Number_of_Patients_sick":getNumSick}
)

# Running all cases that I want to change.
print('loading...\n\n')
batch_run.run_all()
print("done!!")



#%% coarser
run_data = batch_run.get_model_vars_dataframe()

#%%
year = 2022
month = 8
day = 9
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
    'result/batchrun/[{}.{}.{}]CalibrateBeta4.csv'.format(year,month,day))
run_data.to_csv(csv_path)
print("--- %s seconds ---" % (time.time() - start_time))
#%%
data_mean = run_data.groupby(["prob_transmission"])['Number_of_Patients_sick'].mean()
print(data_mean)
print('\n\n')
data_mean = data_mean.reset_index()
print(data_mean['Number_of_Patients_sick'])
mean_patients_sick = data_mean['Number_of_Patients_sick'] #The avg number for the iterations
