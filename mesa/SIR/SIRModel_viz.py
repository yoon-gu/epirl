from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from SIRModel import SIRModel

from scipy import stats
dist_list = [stats.uniform(2,8), stats.norm(loc=5, scale=1)]

def agent_portrayal(agent):
	portrayal = {"Shape": "circle",
				 "Color": "blue",
				 "Filled": "true",
				 "Layer": 2,
				 "r":0.2}
	if agent.status == "Infected":
		portrayal["Color"] = "red"
		portrayal["Layer"] = 0
		portrayal["r"] = 0.5

	elif agent.status == "Recovered":
		portrayal["Color"] = "green"
		portrayal["Layer"] = 1
		portrayal["r"] = 0.5

	return portrayal
grid = CanvasGrid(agent_portrayal, 20, 20, 500, 500)

chart = ChartModule([{"Label":"S", "Color":"Blue"},
					 {"Label":"I", "Color":"Red"},
					 {"Label":"R", "Color":"Green"},
					 {"Label":"V", "Color":"Yellow"},
					 {"Label":"C", "Color":"Orange"},
					 {"Label":"IC", "Color":"Black"}],
					 data_collector_name='datacollector')

model_params = {"N":UserSettableParameter('slider', 'Population', 400, 100, 1000, 100),
				"numOfInfected":UserSettableParameter('slider', 'Initial Infected', 10, 1, 100, 1),
				"infectivity":UserSettableParameter('slider', 'Infectivity', 0.7, 0, 1.0, 0.05),
				"vaccine": UserSettableParameter('slider', 'Daily Vaccine', 0, 0, 100, 10),
				# "recover_day_dist": UserSettableParameter('choice', 'Recover Period Distribution', value='A', choices=['A', 'B']), # only string selection
				"width":20,
				"height":20}

server = ModularServer(SIRModel, [grid, chart], "SIR Model", model_params)

server.port = 8521
server.launch()