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

print(cProfile.run("model.step()","profiling/cpe_cProfiling.prof"))
# Enter the " snakeviz profiling/cpe_cProfiling.prof " in cmd window.

# (1) checkFilled 를 매번 할 필요 없음
# (2) get_cell_list_contents 보다 효율적인 방법
# (3) IsolatedBed 의 step() 은 checkFilled를 두 번 포함 (1)을 해결하면 될 것 같다.
# (4) Nurse 의 step()은 summon 보다 move and spread <-해결방법 고민.