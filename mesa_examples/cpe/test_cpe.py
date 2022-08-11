import pytest
from CPEmodel import CPE_Model
from CPEmodel import height, width

def test_cpe_model():
    numPatients = 30
    numHCW = 4
    numGoo = 3
    model = CPE_Model(num_HCWs=numHCW, num_Patients=numPatients,
                      num_Goo=numGoo, prob_patient_sick=0.01,
                      prob_new_patient=0.05, cleaningDay=40,
                      prob_transmission=.1, isolation_factor=.5,
                      isolate_sick=True, icu_hcw_wash_rate=.9,
                      outside_hcw_wash_rate=.9,
                      height=height, width=width)
