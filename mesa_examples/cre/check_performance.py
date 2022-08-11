from model import CreModel
from tqdm import tqdm

model = CreModel(10, 32, 11)

MaxSteps = 36 * 3 * 24 * 200
for _ in tqdm(range(MaxSteps)):
	model.step()