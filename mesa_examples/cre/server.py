import mesa
from model import CreModel

def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}

    portrayal["Color"] = "grey"
    portrayal["Layer"] = 1
    portrayal["r"] = 0.2

    return portrayal


grid = mesa.visualization.CanvasGrid(agent_portrayal, 10, 10, 500, 500)

model_params = {
    "N": mesa.visualization.Slider(
        "Number of agents",
        100,
        2,
        200,
        1,
        description="Choose how many agents to include in the model",
    ),
    "width": 10,
    "height": 10,
}

server = mesa.visualization.ModularServer(
    CreModel, [grid], "Money Model", model_params
)
server.port = 8521