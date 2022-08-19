from model import CreModel
from tqdm import tqdm

for num_agents in range(10, 110, 20):
	model = CreModel(num_agents, 32, 11)
	MaxSteps = 36 * 3 * 24 * 200
	for _ in tqdm(range(MaxSteps), desc=f'{num_agents} Agents'):
		model.step()