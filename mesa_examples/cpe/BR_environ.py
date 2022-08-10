import mesa
from CPEmodel import CPE_Model, height, width, getNumSick, getHCWInfec
from tqdm import tqdm

if __name__ == '__main__':
    numPatients = 30
    numHCW = 4
    numGoo = 3

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
    results = mesa.batch_run(
        CPE_Model,
        parameters=params,
        iterations=10,
        max_steps=MaxSteps,
        number_processes=8,
        data_collection_period=1,
        display_progress=True,
    )