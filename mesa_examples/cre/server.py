import mesa
from model import CreModel

width = 32
height = 11

def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}

    portrayal["Color"] = "grey"
    portrayal["Layer"] = 1
    portrayal["r"] = 0.2

    return portrayal


grid = mesa.visualization.CanvasGrid(agent_portrayal, width, height, 600, 200)

model_params = {
    "N": mesa.visualization.Slider(
        "Number of agents",
        5,
        2,
        20,
        1,
        description="Choose how many agents to include in the model",
    ),
    "width": width,
    "height": height,
}

server = mesa.visualization.ModularServer(
    CreModel, [grid], "CRE Model", model_params
)
server.port = 8521