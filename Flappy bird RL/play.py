import torch

from model import DQN
from game import FlappyBirdGame

model = DQN()

model.load()

game = FlappyBirdGame()

while True:

    state = torch.tensor(

        game.get_state(),

        dtype=torch.float

    )

    with torch.no_grad():

        prediction = model(state)

    action = torch.argmax(

        prediction

    ).item()

    reward, done, score = game.play_step(action)

    if done:

        print(

            "Score:",

            score

        )

        game.reset()