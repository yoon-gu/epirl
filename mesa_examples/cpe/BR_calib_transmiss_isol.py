from CPEmodel import CPE_Model
from mesa.batchrunner import BatchRunner
import numpy as np
import mesa
import pandas as pd

if __name__ == '__main__':
    # STEP1,STEP2
    num_iter = 10
    runtime = 200 #(Days)

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

    # STEP3
    params = {
        "prob_patient_sick" : probPatientSick,
        "prob_new_patient" : probNewPatient,
        "isolation_factor" : isolationFactor,
        "cleaningDay" : cleanDay,
        "isolate_sick" : isolateSick,
        "isolation_time" : isolationTime,
        "icu_hcw_wash_rate" : ICUwashrate,
        "outside_hcw_wash_rate" : OUTSIDEwashrate,
        "height" : height, "width" : width,
        "prob_transmission" : [0.000001, 0.000002]
        }

    MaxSteps = 36 * 3 * 24 * runtime
    results = mesa.batch_run(
        CPE_Model,
        parameters=params,
        iterations=num_iter,
        max_steps=MaxSteps,
        number_processes=8,
        data_collection_period=1,
        display_progress=True
    )

    run_data = pd.DataFrame(results)

    data_mean = run_data.groupby(["prob_transmission"])['Number of Patients sick'].mean()
    print(data_mean)