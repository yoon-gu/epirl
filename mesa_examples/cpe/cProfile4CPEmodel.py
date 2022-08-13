#%%
import cProfile
from CPEmodel import CPE_Model

probPatientSick = 0.01
probNewPatient = 0.003
probTransmission = 0
isolationFactor = 0.75
isolationTime = 14
cleanDay = 360
isolateSick = True
ICUwashrate = 0.90
OUTSIDEwashrate = 0.90
height=11
width=32

model = CPE_Model(
    prob_patient_sick=probPatientSick, prob_new_patient=probNewPatient,
    prob_transmission=probTransmission,
    isolation_factor=isolationFactor,
    cleaningDay=cleanDay, isolate_sick=isolateSick,
    isolation_time=isolationTime,
    icu_hcw_wash_rate=ICUwashrate,
    outside_hcw_wash_rate=OUTSIDEwashrate,
    height=height, width=width
    )
# %%
print(cProfile.run("model.step()","profiling/cpe_cProfiling.prof"))
# Enter the " snakeviz profiling/cpe_cProfiling.prof " in cmd window.