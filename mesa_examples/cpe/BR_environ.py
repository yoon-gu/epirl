import mesa
from CPEmodel import CPE_Model
from CPEmodel import height, width, getNumSick, getHCWInfec
from mesa.batchrunner import BatchRunner
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
import time

num_iter = 1
runtime = 50 #no. of days

start_time = time.time()
numPatients = 30
numHCW = 4
numGoo = 3
model = CPE_Model(num_HCWs=numHCW, num_Patients=numPatients, num_Goo = numGoo, prob_patient_sick = 0.01, prob_new_patient = 0.05, cleaningDay = 40, prob_transmission = .1, isolation_factor = .5, isolate_sick = True, icu_hcw_wash_rate=.9, outside_hcw_wash_rate=.9, height=height, width=width)


params = {
    "num_HCWs" : numHCW,
    "num_Patients" : numPatients,
    "num_Goo" : 4,
    "prob_patient_sick" : 0.01, # From data
    "prob_new_patient" : 0.003, #0.053, old Calibrated
                                #1/2000, 2592 ticks per day
    "prob_transmission" : 0.0003,
    "isolation_factor" : 0.33,
    "icu_hcw_wash_rate" : .88,
    "outside_hcw_wash_rate" : .67,
    "height": height,
    "width": width,
    "cleaningDay" : [10], "isolate_sick": [0]
}


# 1 day = 24 * 36 * 3 steps
MaxSteps = 24 * 36 * 3 * 350
for _ in tqdm(range(MaxSteps)):
    model.step()

# results = mesa.batch_run(
#     CPE_Model,
#     parameters=params,
#     iterations=1,
#     max_steps=MaxSteps,
#     number_processes=1,
#     data_collection_period=1,
#     display_progress=True,
# )