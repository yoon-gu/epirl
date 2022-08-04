from mesa import Agent
import numpy as np
import random
from routes import length_stay


ICUA = ['A14','A15','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13']
ICUB = ['B14','B15','B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12','B13']

###
def room2coord(room): # Outputs
        
        if len(room) == 2:
            pos = int(room[1]) # number after the letter
        else: # two digit number
            pos = int(room[1])*10 + int(room[2])
        
        letter = room[0] # letter
        if letter == 'A':
            if pos < 12: 
                y = 10
                x = 2 * abs(11-pos) + 5
            elif pos < 14:
                y = 8
                x = 2 * abs(pos-12) + 1
            else:
                y = 8
                x = 2 * abs(pos-14) + 27
        else: # 'B'
            if pos < 12: 
                y = 1
                x = 2 * abs(11-pos) + 5
            elif pos < 14:
                y = 3
                x = 2 * abs(pos-12) + 1
            else:
                y = 3
                x = 2 * abs(pos-14) + 27
        return x,y

class CPE_Agent(Agent):
    """ An agent with fixed colonized status."""
    def __init__(self, unique_id, model, colonized, x, y):
        super().__init__(unique_id, model)
        self.colonized = colonized
        self.isCPE_Agent = True
        self.isPatient = False
        self.isHCW = False
        self.isEnvironment = False
        self.isBed = False
        self.isGoo = False
        self.isWall = False
        self.x = x
        self.y = y


class HCW(CPE_Agent):
    """ An agent with fixed colonized status. Can wash hands"""
    def __init__(self, unique_id, model, colonized, hand_wash_rate, x, y): #path):
        super().__init__(unique_id, model, colonized, x, y)
        self.hand_wash_rate = hand_wash_rate
        
        self.test_rate = 1
        self.true_positive_rate = 1
        
        self.isHCW = True
        self.moveTick = 0
        self.path = []
        self.reached = False
        self.reachedHall = False
        self.hall = 5
    def move(self):
        if self.reached: # reached patient or goo
            self.handwash()
        """
        if self.model.randomMotion:
            notWall = False
        
        
            possible_steps = self.model.grid.get_neighborhood(
                self.pos, moore=False, include_center=False
            )
            #print("Possible steps:", possible_steps)
            
            while not notWall: #while walled
                self.new_pos = self.random.choice(possible_steps)
                #print("New position: ", self.new_pos)
                # Only move if there is no HCW in new_pos
                notWall = True
                new_cellmates = self.model.grid.get_cell_list_contents([self.new_pos])
                for mate in new_cellmates:
                    if mate.isWall:
                        notWall = False
            for mate in new_cellmates:
                if (mate.isHCW) and (mate.new_pos == self.new_pos):
                    self.new_pos = self.pos # stay where you are
            self.model.grid.move_agent(self, self.new_pos)
        
        else: #Nonrandom motion. Go to designated rooms
        """
        # Non random motion by default
        x, y = self.pos
        #print("Self pos: ", self.pos)

        k = len(self.path)
        
        
        #cellmates = self.model.grid.get_cell_list_contents([self.pos])

        destination = room2coord(self.path[self.moveTick])

        # Separate hallways for workers

            
        if (x,y) != destination: # did not find the room yet
            if not self.reachedHall:
                new_position = (x, y + np.sign(self.hall - y))
                self.model.grid.move_agent(self, new_position)
                if y == self.hall:
                    self.reachedHall = True
            else:
                if x < destination[0]:
                    new_position = (x+1, y)
                    self.model.grid.move_agent(self, new_position)  
                elif x > destination[0]:
                    new_position = (x-1, y)
                    self.model.grid.move_agent(self, new_position)  
                elif y > destination[1]:
                    new_position = (x, y-1)
                    self.model.grid.move_agent(self, new_position)  
                elif y < destination[1]:
                    new_position = (x, y+1)
                    self.model.grid.move_agent(self, new_position)
            """go to col 5
            then go hori.
            Then go up or down till reached.
            """
        else: # found
            self.moveTick += 1
            self.moveTick %= k
            self.reachedHall = False
            self.move() # recursion, so that it will continue on asap.

    def testCRE(self, other):
        test = np.random.choice([1,0], p = [self.test_rate, 1-self.test_rate])
        if test:
            positive = np.random.choice([1,0], p = [self.true_positive_rate, 1-self.true_positive_rate])
            if positive:
                other.positive = True

    def spread(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        self.reached = False
        if len(cellmates)>1:

            #other = self.random.choice(cellmates)
            # so if uncolonized doctor and patient are in same square, only one gets colonized
            for other in cellmates:
                if ((other.isPatient) or (other.isGoo)): #only infec/get infected by Goo or Patients, NOT BEDS
                    prob_transmission = self.model.prob_transmission
                    if self.colonized:
                        
                        # If we want to separate goo and patient, uncomment the following code
                        """if (other.isPatient and not other.colonized):"""
                        if other.isolated: #other.isPatient and other.isolated: #for isolated patients, decrease by a factor
                            prob_transmission *= self.model.isolation_factor    
                        infect = np.random.choice([1,0], p = [prob_transmission, 1-prob_transmission]) # note that we did not use self.model.prob_transmission

                        if infect == 1:
                            if other.isPatient and not other.colonized: # nonsick patient
                                other.colonized = True
                                other.stay += 7 #lengthen the stay
                                self.model.cumul_sick_patients += 1
                                self.model.num_infecByHCW += 1    
                            else: # (other.isGoo):
                                other.colonized = True
                    
                    else: #get infected
                        if other.colonized:
                            if other.isolated:
                                prob_transmission *= self.model.isolation_factor
                            infect = np.random.choice([1,0], p = [prob_transmission, 1-prob_transmission]) # note that we did not use self.model.prob_transmission
                            self.colonized = True
                            
                            """check if sick"""
                            if other.isPatient:
                                self.testCRE(other)
                
                if other.isPatient or other.isGoo:
                    self.reached = True
                #else:
                #   self.reached = False

    def handwash(self):
        wash = np.random.choice([1,0], p = [self.hand_wash_rate, 1-self.hand_wash_rate]) # fixed proability based on data       
        if wash:
            self.colonized = False
class Nurse(HCW):
    def __init__(self, unique_id, model, colonized, hand_wash_rate, x, y): #path):
        super().__init__(unique_id, model, colonized, hand_wash_rate, x, y)
        self.workHours = 8
        
    def step(self):
        self.move()
        self.spread()
        
        # Change nurses. Handwash.
        if self.model.tick == self.model.ticks_in_hour * self.workHours:
            self.colonized = False
            
class Dr(HCW):
    def __init__(self, unique_id, model, colonized, hand_wash_rate, x, y, workHours, numCare): #path):
        super().__init__(unique_id, model, colonized, hand_wash_rate, x, y)
        self.original_pos = (x,y)
        self.workHours = workHours
        self.activated = False
        self.path = random.sample(ICUA+ICUB, numCare)
        self.startTime = 0
        self.endTime = self.startTime + self.model.ticks_in_hour * self.workHours

    def leaveICU(self):
        destination = self.original_pos
        if self.pos != destination: # original position
            x, y = self.pos
            if not self.reachedHall:
                new_position = (x, y + np.sign(self.hall - y))
                self.model.grid.move_agent(self, new_position)
                if y == self.hall:
                    self.reachedHall = True
            else: # reachedHall
                if x < destination[0]:
                    new_position = (x+1, y)
                    self.model.grid.move_agent(self, new_position)  
                elif x > destination[0]:
                    new_position = (x-1, y)
                    self.model.grid.move_agent(self, new_position)  
                elif y > destination[1]:
                    new_position = (x, y-1)
                    self.model.grid.move_agent(self, new_position)  
                elif y < destination[1]:
                    new_position = (x, y+1)
                    self.model.grid.move_agent(self, new_position)
            
        else: # found
            self.reachedHall = False
            pass # get some rest
    
    def step(self):
        if self.model.tick > self.startTime and self.model.tick < self.endTime:
            self.activated = True
        elif self.model.tick == self.endTime:
            self.activated = False
            self.colonized = False
            
            
        if self.activated:
            self.move()
            self.spread()
        else: #deactivated
            self.leaveICU()

class XrayDr(Dr):
    def __init__(self, unique_id, model, colonized, hand_wash_rate, x, y, workHours, numCare, shiftsPerDay):
        super().__init__(unique_id, model, colonized, hand_wash_rate, x, y, workHours, numCare)
        self.shiftsPerDay = shiftsPerDay    
        self.original_pos = (x,y)
        self.activated = False
        self.path = random.sample(ICUA+ICUB, numCare)
        self.k = len(self.path)
        self.last_patient = self.path[numCare-1]
    
    
    def step(self):
        if self.model.tick % (self.model.ticks_in_day / self.shiftsPerDay)== 0:
            self.activated = True
        if self.pos == room2coord(self.last_patient):
            self.activated = False
            self.reachedHall = False
            self.moveTick = 0
        
        if self.activated:
            self.move()
            self.spread()
        else:
            self.leaveICU()
        
class Patient(CPE_Agent):
    """ An agent with colonized status.
    Cannot move and leaves hospital after a certain time"""
    def __init__(self, unique_id, model, colonized, x, y):
        super().__init__(unique_id, model, colonized, x, y)
        self.isPatient = True
        #self.infecByHCW = False # in the beginning, nobody is infected by HCW
        
        self.checkIsolated()
        self.model.current_patients.append(self)
        #self.state
        self.state = 'healthy'
        sick = np.random.choice([1,2], p=[self.model.prob_patient_sick,1-self.model.prob_patient_sick])
        if sick == 1:
            self.colonized = True
            self.state = 'sick'

        
        if self.colonized: #mean 11, 1q 3.5,  median 6, 3q 16.75
            self.stay = np.random.geometric(.1)*self.model.ticks_in_day
            
        else: #mean 7 median 2, 1q 0, 3q 9
            # right skewed, so use geometric random var
            self.stay = np.random.geometric(.2)*self.model.ticks_in_day
        
        #self.new = True # SCV ready to go sir

        self.positive = False
        
    def checkIsolated(self):
        if self.x <= 7 or self.x >= 23:
            self.isolated = True
        else:
            self.isolated = False

    def step(self):
        #remove oneself if the stay is too long
        self.stay -= 1
        if self.stay == 0:
            self.model.current_patients.remove(self)
            self.model.discharged.append(self)
        
        #if self.new == True:
        #    print("New Patient {}: {}, {} days left".format(self.unique_id, self.state, self.stay))
        #else:
        #    print("Patient {}: {}, {} days left".format(self.unique_id, self.state, self.stay))
        #self.new = False

        self.checkIsolated() #alters state if moved to/from isolated bed
    
    
#%%


class Environment(CPE_Agent):
    """ An agent with colonized status.
    Cannot move and does not leave."""


    def __init__(self, unique_id, model, colonized, x, y):
        super().__init__(unique_id, model, colonized, x, y)
        self.isEnvironment = True    
        self.filled = False # all patients fill the beds
        self.filledSick = False
        self.isGoo = False
        self.isBed = False
        self.isIsolatedBed = False
        self.isWall = False
        



class Wall(Environment):
    """ An agent with colonized status.
    HCW cannot pass through walls."""
    def __init__(self, unique_id, model, colonized, x, y):
        super().__init__(unique_id, model, colonized, x, y)
        self.isWall = True
    def step(self):
        pass

class Bed(Environment):
    """ An agent with colonized status.
    Can get contaminated, but cleans after a patient leaves."""
    def __init__(self, unique_id, model, colonized, x, y):
        super().__init__(unique_id, model, colonized, x, y)
        self.isBed = True
        self.isIsolatedBed = False

    def step(self):
        self.checkFilled()
        if not self.filled:
            incoming = np.random.choice([1,0], p = [self.model.prob_new_patient, 1-self.model.prob_new_patient])
            if (incoming == 1):
                new_patient = Patient(self.model.num_Patients + self.model.cumul_patients, self.model, colonized = False, x = self.x, y = self.y)
                self.model.schedule.add(new_patient)
                self.model.grid.place_agent(new_patient, (self.x, self.y))
                self.model.cumul_patients += 1
                if new_patient.colonized:
                    self.model.cumul_sick_patients += 1
        if not self.filled:
            self.model.empty_beds.add((self.x, self.y))
        else:
            if (self.x, self.y) in self.model.empty_beds: # without checking element in set, there could be error. 
                self.model.empty_beds.remove((self.x, self.y)) 
        self.checkFilled()
        # if self.filled == False:
        #     self.colonized = False
    def checkFilled(self):
        self.filled = False
        self.filledSick = False #reset
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates)>1:
            for cellmate in cellmates:
                if cellmate.isPatient:
                    self.filled = True
                    if self.filled and cellmate.positive:
                        self.filledSick = True
 
                    break # other contents of bed doesn't matter
        



class IsolatedBed(Bed):
    """ An agent with colonized status.
    Can get contaminated, but cleans after a patient leaves.
    The isolated bed has lower probability of transmission compared to other beds."""
    def __init__(self, unique_id, model, colonized, x, y):
        super().__init__(unique_id, model, colonized, x, y)
        self.isIsolatedBed = True
        
    def step(self):
        self.checkFilled()
        if not self.filled:
            incoming = np.random.choice([1,0], p = [self.model.prob_new_patient, 1-self.model.prob_new_patient])
            if (incoming == 1):
                new_patient = Patient(self.model.num_Patients + self.model.cumul_patients, self.model, colonized = False, x = self.x, y = self.y)
                self.model.schedule.add(new_patient)
                self.model.grid.place_agent(new_patient, (self.x, self.y))
                self.model.cumul_patients += 1
                if new_patient.colonized:
                    self.model.cumul_sick_patients += 1
        if not self.filled:
            self.model.empty_beds.add((self.x, self.y))
        else: # not empty anymore!
            if (self.x, self.y) in self.model.empty_beds: # without checking element in set, there could be error. 
                self.model.empty_beds.remove((self.x, self.y)) 
        self.checkFilled()
        #print("Bed {}. Filled: {}, FilledSick: {}".format(self.unique_id, self.filled, self.filledSick))
        
        #if self.filled == False:
        #    self.colonized = False

class Goo(Environment):
    """ An agent with colonized status.
    Can get contaminated, cleans once in a while."""
    def __init__(self, unique_id, model, colonized, x, y):
        super().__init__(unique_id, model, colonized, x, y)
        self.isGoo = True
        self.clean_tick = self.model.cleaningDay * self.model.ticks_in_day
        
        if self.x <= 7 or self.x >= 23:
            self.isolated = True
        else:
            self.isolated = False
    def handwash(self): # 100%
        #wash = np.random.choice([1,0], p = [.9, .1]) # fixed proability based on data       
        #if wash:
        self.colonized = False
        
    def step(self):
        #self.checkFilled()
        self.clean_tick -= 1
        
        if (self.clean_tick <= 0):
            self.handwash()
            self.clean_tick = self.model.cleaningDay * self.model.ticks_in_day
        
    

# make a patient class, test run