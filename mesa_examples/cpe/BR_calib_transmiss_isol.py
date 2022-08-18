#%%
from CPEmodel import CPE_Model
from CPEmodel import getNumSick, getHCWInfec
from mesa.batchrunner import BatchRunner

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import mesa
import os
import pandas as pd
import time
import seaborn as sns

if __name__ == '__main__':
    # %% STEP1,STEP2
    num_iter = 10
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
    params = {
        "prob_patient_sick" : probPatientSick,
        "prob_new_patient" : probNewPatient,
        # "prob_transmission" : probTransmission,
        "isolation_factor" : isolationFactor,
        "cleaningDay" : cleanDay,
        "isolate_sick" : isolateSick,
        "isolation_time" : isolationTime,
        "icu_hcw_wash_rate" : ICUwashrate,
        "outside_hcw_wash_rate" : OUTSIDEwashrate,
        "height" : height, "width" : width,
        "prob_transmission" : [0.000001, 0.000002]
        }

    # Specify the variable I want to change separately.
    model = CPE_Model(
        prob_patient_sick=probPatientSick,prob_new_patient=probNewPatient, prob_transmission=probTransmission,
        isolation_factor=isolationFactor,cleaningDay=cleanDay, isolate_sick=True, isolation_time=isolationTime,
        icu_hcw_wash_rate=ICUwashrate, outside_hcw_wash_rate=OUTSIDEwashrate,
        height=height, width=width
        )

    MaxSteps = model.ticks_in_day * runtime
    results = mesa.batch_run(
        CPE_Model,
        parameters=params,
        iterations=10,
        max_steps=MaxSteps,
        number_processes=8,
        data_collection_period=1,
        display_progress=True
    )

    # Running all cases that I want to change.
    print('loading...\n\n')
    batch_run.run_all()
    print("done!!")


    #%% coarser
    run_data = batch_run.get_model_vars_dataframe()

    #%%
    print("--- %s seconds ---" % (time.time() - start_time))
    #%%
    data_mean = run_data.groupby(["prob_transmission"])['Number_of_Patients_sick'].mean()
    print(data_mean)
    print('\n\n')
    data_mean = data_mean.reset_index()
    print(data_mean['Number_of_Patients_sick'])
    mean_patients_sick = data_mean['Number_of_Patients_sick'] #The avg number for the iterations
