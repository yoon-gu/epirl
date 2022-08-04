#%%
"""CPEmodel.py goes with cpe_Vis, so visitPatient() has been deleted"""
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import BaseScheduler, RandomActivation, SimultaneousActivation
import numpy as np
import matplotlib.pyplot as plt
import random
from mesa.space import MultiGrid
from numpy.lib.shape_base import expand_dims #add a location variable
#constants

from agents import *

#%%

width = 32
height = 11
numPatients = 30
numHCW = 2
numGoo = 4
def getTotPatients(model):
    count = 0
    l = [agent.isPatient for agent in model.schedule.agents]
    for i in l:
        if i: # i is patient
            count += 1
    return count

def getNumSick(model):
    """for patients"""
    count = 0
    l = [(agent.isPatient and agent.colonized) for agent in model.schedule.agents]
    for i in l:
        if i: # i is patient
            count += 1
    return count


def getNumColonized(model):
    """for HCW"""
    count = 0
    l = [(agent.isHCW and agent.colonized) for agent in model.schedule.agents]
    for i in l:
        if i:
            count += 1
    return count

def getCumulNumSick(model):
    """cumulative sick patients"""
    
    return model.cumul_sick_patients

def getHCWInfec(model):
    """every time a HCW infects a patient"""

    return model.num_infecByHCW

def getCumul(model):
    """cumulative patients"""
    return model.cumul_patients


#%%
ICUA = ['A14','A15','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13']
ICUB = ['B14','B15','B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12','B13']

"""if we want to shuffle"""
#random.shuffle(ICUA)
#random.shuffle(ICUB)
import numpy as np
path_A = np.array_split(ICUA, 5)
path_B = np.array_split(ICUB, 5)
paths = np.concatenate((path_A, path_B))
#path_A1 = random.sample(ICUA, 7)
#path_A2 = [x for x in ICUA if x not in path_A1]
#path_B1 = random.sample(ICUB, 7)
#path_B2 = [x for x in ICUB if x not in path_B1]
#paths = [path_A1, path_A2, path_B1, path_B2]

class CPE_Model(Model):
    """A model with some number of agents."""
    def __init__(self,  num_HCWs, num_Patients, num_Goo, prob_patient_sick, prob_new_patient, prob_transmission, isolation_factor, cleaningDay, isolate_sick, icu_hcw_wash_rate, outside_hcw_wash_rate, height, width):
        self.tick = 0
        self.ticks_in_hour = 36 * 3 # 36 ticks to visit 3 patients, 3 cycles per hour
        self.ticks_in_day = 24 * self.ticks_in_hour
        self.num_HCWs = num_HCWs
        self.num_Patients = num_Patients
        self.num_Goo = num_Goo
        self.schedule = BaseScheduler(self)
        self.grid = MultiGrid(width, height, torus =False)
        self.prob_patient_sick = prob_patient_sick
        self.prob_new_patient = prob_new_patient # geometric rv
        self.cleaningDay = cleaningDay 
        self.isolate_sick = isolate_sick
        self.prob_transmission = prob_transmission
        self.isolation_factor = isolation_factor
        self.discharged = []
        self.current_patients = [] # List of all patients
        self.empty_beds = set() # empty set
        self.shared_beds = [] # list of all non isolated beds
        self.isol_beds = [] # list of all isolated beds
        self.isol_space = False # Is there space in the isolated rooms?
        self.num_infecByHCW = 0
        self.icu_hcw_wash_rate = icu_hcw_wash_rate
        self.outside_hcw_wash_rate = outside_hcw_wash_rate
        self.summon = False
        self.summoner = 0
        self.nurse_list = []
        
        #self.randomMotion = randomMotion # or False for circular motion

        #cumulative patients
        self.cumul_patients = num_Patients
        
        #cumulative sick patients
        self.cumul_sick_patients = 0

        # Create Xray Drs
        strange = XrayDr(-self.num_HCWs - 7, self, colonized = False, hand_wash_rate = self.outside_hcw_wash_rate, x = width-1, y = 0, workHours=24, numCare = 30, shiftsPerDay = 2)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))
        
        strange = XrayDr(-self.num_HCWs - 8, self, colonized = False, hand_wash_rate = self.outside_hcw_wash_rate, x = width-2, y = 0, workHours=24, numCare = 20, shiftsPerDay = 1)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))
        
        strange = XrayDr(-self.num_HCWs - 9, self, colonized = False, hand_wash_rate = self.outside_hcw_wash_rate, x = width-2, y = 1, workHours=24, numCare = 3, shiftsPerDay = 1)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))
        
        
        
        # Drs
        
        strange = Dr(-self.num_HCWs - 6, self, colonized = False, hand_wash_rate = self.icu_hcw_wash_rate, x = width-1, y = height-1, workHours=24, numCare = 30)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))
        
        strange = Dr(-self.num_HCWs - 5, self, colonized = False, hand_wash_rate = self.icu_hcw_wash_rate, x = width-2, y = height-1, workHours=12, numCare = 30)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))
        strange = Dr(-self.num_HCWs - 4, self, colonized = False, hand_wash_rate = self.icu_hcw_wash_rate, x = width-2, y = height-2, workHours=12, numCare = 30)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))
        
        strange = Dr(-self.num_HCWs - 3, self, colonized = False, hand_wash_rate = self.icu_hcw_wash_rate, x = width-1, y = height-2, workHours=3, numCare = 30)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))
        
        strange = Dr(-self.num_HCWs - 2, self, colonized = False, hand_wash_rate = self.icu_hcw_wash_rate, x = width-4, y = height-1, workHours=1, numCare = 30)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))
        strange = Dr(-self.num_HCWs - 1, self, colonized = False, hand_wash_rate = self.icu_hcw_wash_rate, x = width-4, y = height-2, workHours=1, numCare = 30)
        self.schedule.add(strange)
        self.grid.place_agent(strange, (strange.x, strange.y))

        
        
        
        
        
        # Create HCW agents
        for j in range(-self.num_HCWs, 0):
            b = Nurse(j, self, colonized = False, hand_wash_rate = self.icu_hcw_wash_rate, x = -2*j-2, y = 0)#, path = ex_path)
            self.schedule.add(b)
            self.grid.place_agent(b, (b.x, b.y)) #added 1 to start near patients for route simulation (temporary solution)
            #print(model.schedule.agents)
            b.path = paths[-j-1] 
            if b.path[0][0] == 'A':
                b.hall = 6 #ICUA
            else:
                b.hall = 5 #ICUB
            self.nurse_list.append(b)
            
        # bed
        for i in range(30):
            
            
            if i in range(11):
                xpos = 2 * i + 5
                ypos = 1
            elif i in range(11, 13):
                #isolated = True
                xpos = 2 * (i-11) + 1
                ypos = 3
            elif i in range(13, 15):
                #isolated = True
                xpos = 2 * (i-13) + 27
                ypos = 3
            elif i in range(15, 17):
                #isolated = True
                xpos = 2 * (i-15) + 1
                ypos = 8
            elif i in range(17, 19):
                #isolated = True
                xpos = 2 * (i-17) + 27
                ypos = 8
            elif i in range(19, 30):
                xpos = 2 * (i-19) + 5
                ypos = 10
            
            #isolated status
            isolated = False
            if xpos in {1,3,5,7,23,25,27,29}: #isolation
                isolated = True
            
            """We first create beds so that patient.checkIsolated() works"""
            if isolated:
                b = IsolatedBed(i, self, colonized = False, x = xpos, y = ypos) # 0~30
                self.schedule.add(b)
                self.grid.place_agent(b, (b.x, b.y))
                self.isol_beds.append(b)
            else:
                b = Bed(i, self, colonized = False, x = xpos, y = ypos) # 0~30
                self.schedule.add(b)
                self.grid.place_agent(b, (b.x, b.y))
                self.shared_beds.append(b)
            
            #Patients
            a = Patient(self.num_Patients + i, self, colonized = False, x = xpos, y = ypos)#30~60
            self.schedule.add(a)
            self.grid.place_agent(a, (a.x, a.y))
            if a.colonized == True:
                self.cumul_sick_patients += 1
                a.stay = np.random.randint(1,17+1) * self.ticks_in_day
            else:
                a.stay = np.random.randint(1,9+1) * self.ticks_in_day
            
            
            # Goo
            if ypos > 5: # top row
                d = Goo(-self.num_Patients - i, self, colonized = True, x=xpos, y=ypos-1) #-30~-60
            else: # bottom row
                d = Goo(-self.num_Patients - i, self, colonized = True, x=xpos, y=ypos+1) #-30~-60
            self.schedule.add(d)
            self.grid.place_agent(d, (d.x, d.y))



        """
        The Great Wall
        # for j in range(2,20):
        #     if j not in range(10,13):
        #         b = Wall(-100-j, self, colonized = False, x = j, y = 5)
        #         #self.schedule.add(b)
        #         self.grid.place_agent(b, (b.x, b.y))

        # for j in range(-num_Goo-num_HCWs, -num_HCWs):
        #     tempx = self.random.randrange(self.grid.width)
        #     tempy = self.random.randrange(self.grid.height)
        #     same_as_bed = False
        #     for k in range(self.num_Patients):
        #         if (2*(k%6)+1,(k//6)*3+1) == (tempx, tempy):
        #             same_as_bed = True
        #     while same_as_bed == True:
        #         tempx = self.random.randrange(self.grid.width)
        #         tempy = self.random.randrange(self.grid.height)
        #         same_as_bed = False
        #         for k in range(self.num_Patients):
        #             if (2*(k%6)+1,(k//6)*3+1) == (tempx, tempy):
        #                 same_as_bed = True
        """



        self.datacollector = DataCollector(
            model_reporters={"Number of Patients sick": getNumSick,
                    "Number of HCW colonized": getNumColonized,
                    "Cumulative number of colonized patients": getCumulNumSick,
                    "HCW related infections": getHCWInfec,
                    "Cumulative Patients": getCumul,
                    #"Total number of Patients": getTotPatients
                    })
            
        self.running = True
        self.datacollector.collect(self)

    def get_nurse_list(self):
        l = []
        for m in model.schedule.agents:
            if isinstance(m, Nurse):
                l.append(m)
        return l

    def get_patient_list(self):
        l = []
        for m in model.schedule.agents:
            if m.isPatient:
                l.append(m)
        return l

    def step(self):
        #print('Cumulative Patients: {}'.format(self.cumul_patients))

        self.tick += 1
        self.tick %= self.ticks_in_day # to keep the number from getting too large
        if self.tick % self.ticks_in_hour == 0:
            self.summoner = self.tick // self.ticks_in_hour
            if self.summoner > 0 and self.summoner < 16: # only for patients 1~15
                self.summon = True

        self.schedule.step()
        self.datacollector.collect(self)
        for ex_patient in self.discharged:
            # get location of the patient to be discharged
            tempx = ex_patient.x
            tempy = ex_patient.y
            
            # remove patient
            #self.remove_patient(ex_patient)
            self.grid.remove_agent(ex_patient)
            self.schedule.remove(ex_patient)
            self.discharged.remove(ex_patient)

            # add a new patient in place of old patient
            #def __init__(self, unique_id, model, colonized, x, y):
        
        #Instead of prob for each bed, make it into prob for entire ward
        """if not not self.empty_beds: #if empty_beds is not empty
            incoming = np.random.choice([1,0], p = [self.prob_new_patient, 1-self.prob_new_patient])
            if (incoming == 1):
                bed = self.empty_beds.pop() # 
                tempx, tempy = bed # unpack the tuple
                new_patient = Patient(self.num_Patients + self.cumul_patients, self, colonized = False, x = tempx, y = tempy)
                self.schedule.add(new_patient)
                self.grid.place_agent(new_patient, (tempx, tempy))
                self.cumul_patients += 1
                if new_patient.colonized == True:
                    self.cumul_sick_patients += 1
        """
        
        """Move patients to Isolated beds"""
        if self.isolate_sick:
            for bed in self.shared_beds: # 7-person room
                if bed.filledSick:
                    cellmates = self.grid.get_cell_list_contents([bed.pos])
                    for cellmate in cellmates: # in case HCW is also in the same cell
                        
                        if not cellmate.isPatient:
                            continue # hcw or THE BED ITSELF
                        
                        sickguy = cellmate
                        for ibed in self.isol_beds:
                            if ibed.filledSick:
                                continue
                            if ibed.filled:
                                icellmates = self.grid.get_cell_list_contents([ibed.pos])
                                for icellmate in icellmates:
                                    if icellmate.isPatient:
                                        healthyguy = icellmate
                                        self.grid.move_agent(sickguy, (ibed.x, ibed.y))
                                        self.grid.move_agent(healthyguy, (bed.x, bed.y))
                                        ibed.checkFilled() # to label the bed filled
                                        break # we don't need to consider other icellmates
                                break # we don't need to consider other beds
                            else: # not filled
                                self.grid.move_agent(sickguy, (ibed.x, ibed.y))
                                ibed.checkFilled() # to label the bed filled
                                break
                        
                else:
                    continue




    def run_model(self, n):
        for i in range(n):
            self.step()

    #print('\n')


# %%
cleaningDay = 40 # every 40 days
model = CPE_Model(num_HCWs=numHCW, num_Patients=numPatients, num_Goo = numGoo, prob_patient_sick = 0.5, prob_new_patient = 0.5, cleaningDay = cleaningDay, prob_transmission = .9, isolation_factor = .5, isolate_sick = True, icu_hcw_wash_rate=.9, outside_hcw_wash_rate=.9, height=height, width=width)

# %%
