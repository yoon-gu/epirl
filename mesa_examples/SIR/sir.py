from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
from scipy import stats

class PersonAgent(Agent):
	"""An agent with fixed initial wealth."""
	def __init__(self, unique_id, model, status="Susceptible", infectivity=0.7, recover_day=5):
		super(PersonAgent, self).__init__(unique_id, model)
		self.day_after_infection = 0
		self.infectivity = infectivity
		self.recover_day = recover_day
		self.status = status
		self.success_infection = 0

	def step(self):
		"""The agent's step will go here"""
		self.move()
		if self.status == "Infected":
			if self.day_after_infection > self.recover_day:
				self.status = "Recovered"
			self.day_after_infection += 1

		self.talk()

	def move(self):
		possible_steps = self.model.grid.get_neighborhood(
			self.pos, moore=True, include_center=False)
		new_position = self.random.choice(possible_steps)
		self.model.grid.move_agent(self, new_position)

	def talk(self):
		cellmates = self.model.grid.get_cell_list_contents([self.pos])
		if self.status == "Infected":
			for other in cellmates:
				if other.status == "Susceptible":
					self.model.num_contact_of_infected += 1
					if np.random.uniform(0, 1, size=1) < self.infectivity:
						self.success_infection += 1
						other.status = "Infected"

class SIRModel(Model):
	"""A model with some number of agents"""
	def __init__(self, N, width, height, numOfInfected, infectivity=0.7, vaccine=0,
				recover_day_dist=stats.uniform(3, 7)):
		self.num_agents = N
		self.numOfInfected = numOfInfected
		self.grid = MultiGrid(width, height, True)
		self.schedule = RandomActivation(self)
		self.running = True
		self.vaccine = vaccine
		self.infectivity = infectivity
		self.num_contact_of_infected = 0
		self.incidence = 0
		# self.initial_infection_agent = self.random.randint(N, size=(numOfInfected))

		# Create agents
		for i in range(self.num_agents):
			status = "Susceptible"
			if i < self.numOfInfected:
				status = "Infected"
			a = PersonAgent(i, self, status, self.infectivity, recover_day_dist.rvs())
			self.schedule.add(a)

			# add the agent to a random grid cell
			x = self.random.randrange(self.grid.width)
			y = self.random.randrange(self.grid.height)
			self.grid.place_agent(a, (x, y))

		self.datacollector = DataCollector(
			model_reporters={
				"S": lambda m: self.count_type(m, "Susceptible"),
				"I": lambda m: self.count_type(m, "Infected"),
				"R": lambda m: self.count_type(m, "Recovered"),
				"V": lambda m: self.count_type(m, "Vaccined"),
				"C": lambda m: self.num_contact_of_infected,
				"IC": lambda m: self.incidence
			})

		self.running = True
		self.datacollector.collect(self)

	def step(self):
		"""Advance the model by one step"""
		S = [agent for agent in self.schedule.agents if agent.status == "Susceptible"]
		self.incidence = len([agent for agent in self.schedule.agents if (agent.status == "Infected") and agent.day_after_infection == 0])
		vaccined_population = np.random.choice(S, min(self.vaccine, len(S)), replace=False)
		for agent in vaccined_population:
			agent.status = "Vaccined"
		self.schedule.step()
		self.datacollector.collect(self)
		if self.count_type(self, "Infected") == 0:
			self.running = False

	@staticmethod
	def count_type(model, status):
		count = 0
		for agent in model.schedule.agents:
			if agent.status == status:
				count += 1
		return count