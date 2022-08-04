#%%
from CPEmodel import CPE_Model
from CPEmodel import height, width, getNumSick, getHCWInfec
from mesa.batchrunner import BatchRunner

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
import time

#%% 
num_iter = 30
runtime = 300 #no. of days

start_time = time.time()
numPatients = 30
numHCW = 4
numGoo = 3
model = CPE_Model(num_HCWs=numHCW, num_Patients=numPatients, num_Goo = numGoo, prob_patient_sick = 0.01, prob_new_patient = 0.05, cleaningDay = 40, prob_transmission = .1, isolation_factor = .5, isolate_sick = True, icu_hcw_wash_rate=.9, outside_hcw_wash_rate=.9, height=height, width=width)


fixed_params = {
    "num_HCWs" : numHCW,
    "num_Patients" : numPatients,
    "num_Goo" : 4,
    "prob_patient_sick" : 0.01, # From data
    "prob_new_patient" : 0.003, #0.053, old Calibrated
                                #1/2000, 2592 ticks per day
    #"cleaningDay": 40,
    #"isolate_sick" : False,
    "prob_transmission" : 0.0003,
    "isolation_factor" : 0.33,
    "icu_hcw_wash_rate" : .88,
    "outside_hcw_wash_rate" : .67,
    "height": height,
    "width": width,
}

#xs =  np.arange(.06, .08, .005)#(.03, .13, .03) #np.arange(.02, .96, 0.02)
#ys =  np.arange(.52, 1.1, .12)#np.arange(.6, .91, 0.1)#np.arange(.05, .6, 0.15)

#variable_params = {"prob_transmission" : xs, "isolation_factor" : ys}

zs = [10, 30, 60, 100, 180]
ws = [0, 1]
variable_params = {"cleaningDay" : zs, "isolate_sick": ws}
batch_run = BatchRunner(
    CPE_Model,
    variable_params,
    fixed_params,
    iterations = num_iter,
    max_steps = model.ticks_in_day * runtime,
    #model_reporters = {"Number_of_Patients_sick":getNumSick}
    model_reporters = {"HCW_related_infecs": getHCWInfec}
)

batch_run.run_all()
#%%
run_data = batch_run.get_model_vars_dataframe()
data_mean = run_data.groupby(["cleaningDay", "isolate_sick"])['HCW_related_infecs'].mean()
print(data_mean)
print('\n\n')
data_mean = data_mean.reset_index()
print(data_mean['HCW_related_infecs'])
mean_patients_sick = data_mean['HCW_related_infecs'] #The avg number for the iterations
#%%
data_mean.columns
data_mean = data_mean.rename(columns = {'cleaningDay':'isolate_sick', 'isolate_sick':'cleaningDay'})

#%%
w = .4
#data_mean2 = data_mean.rename(columns={"isolate_sick": "cleaningDay", "cleaningDay":"isolate_sick"})
isolated = data_mean.loc[data_mean['isolate_sick']==1]
nonisol = data_mean.loc[data_mean['isolate_sick']==0]
#%%
zz = np.arange(len(zs))
zi = [i + w for i in np.arange(len(zs))]

z = list(map(str, zs))#[0:-1]
#z.append("Never")
#%%
fig = plt.figure()
plt.bar(zz, nonisol["HCW_related_infecs"], w, label = "No Isolation")
plt.bar(zi, isolated["HCW_related_infecs"], w, label = "Isolation")
plt.xlabel("Days before cleaning environment")
plt.ylabel("Number of HCW related infections")
plt.xticks(zz + w/2, z)
plt.title("HCW related infections after {} days ({} iterations)".format(runtime, num_iter))
plt.legend()
#%%
plt.show()
#%%
"""
#%%

data = dict(isolated)
fig = plt.figure()
#fig.suptitle("HCW related infections after {} days ({} iterations)".format(runtime, num_iter))
#ax = plt.axes(projection='3d')
#ax.scatter(xv,yv,mean_patients_sick, c = mean_patients_sick, cmap = 'plasma')
#plt.bar(list(zs), isolated['HCW_related_infecs'])
z = list(map(str, zs))[0:-1]
z.append("Never")
plt.bar(z, list(data.values()))
plt.xlabel("Days before cleaning environment")
plt.ylabel("Number of HCW related infections")
plt.title("HCW related infections after {} days ({} iterations)".format(runtime, num_iter))
plt.show()

plt.xlabel("Days before cleaning environment")
plt.ylabel("Number of HCW related infections")
plt.title("HCW related infections after {} days ({} iterations)".format(runtime, num_iter))
plt.show()
#%%

""""""
#%%
#xv, yv = np.meshgrid(xs, ys)
#elev = 25 # good angle for graph
#azim = 35-180 # good angle for graph

##data = {'Isolation': mean_patients_sick[1], 'No Isolation': mean_patients_sick[0]}
data = dict(mean_patients_sick)
fig = plt.figure()
#fig.suptitle("HCW related infections after {} days ({} iterations)".format(runtime, num_iter))
#ax = plt.axes(projection='3d')
#ax.scatter(xv,yv,mean_patients_sick, c = mean_patients_sick, cmap = 'plasma')
plt.bar(list(zs), list(data.values()))

plt.xlabel("Days before cleaning environment")
plt.ylabel("Number of HCW related infections")
plt.title("HCW related infections after {} days ({} iterations)".format(runtime, num_iter))
plt.show()
#ax.set_xlabel('Transmission Probability')
# ax.set_ylabel('Isolation Factor')
#ax.set_zlabel('Patients sick')
#ax.view_init(elev, azim)

#%%
fig = plt.figure()
z = list(map(str, zs))[0:-1]
z.append("Never")
plt.bar(z, list(data.values()))
plt.xlabel("Days before cleaning environment")
plt.ylabel("Number of HCW related infections")
plt.title("HCW related infections after {} days ({} iterations)".format(runtime, num_iter))
plt.show()
"""
#%%
image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
    'result\\batchrun\\environ_isol300.png')

fig.savefig(image_path)
#%%
# csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
#     'result\\batchrun\\trans_isol.csv')
# run_data.to_csv(csv_path)
# print("--- %s seconds ---" % (time.time() - start_time))
#%%
plt.show()
# %%
