import cProfile
from CPEmodel import CPE_Model
from mesa.batchrunner import BatchRunner
from CPEmodel import getNumSick

num_iter = 1
runtime = 200

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
beta = [0.00001, 0.00002]
variable_params = {"prob_transmission" : beta}

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

batch_run = BatchRunner(
    CPE_Model,
    variable_parameters = variable_params,
    fixed_parameters = fixed_params,
    iterations=num_iter,
    max_steps=model.ticks_in_day * runtime,
    model_reporters = {"Number_of_Patients_sick":getNumSick}
)

print(cProfile.run("batch_run.run_all()","profiling/BR_cProfiling.prof"))
# Enter the " snakeviz profiling/BR_cProfiling.prof " in cmd window.
