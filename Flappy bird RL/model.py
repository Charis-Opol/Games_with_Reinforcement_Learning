import os

import torch

import torch.nn as nn

import torch.optim as optim


class DQN(nn.Module):

    def __init__(

        self,

        input_size=5,

        hidden_size=128,

        output_size=2

    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(

                input_size,

                hidden_size

            ),

            nn.ReLU(),

            nn.Linear(

                hidden_size,

                hidden_size

            ),

            nn.ReLU(),

            nn.Linear(

                hidden_size,

                output_size

            )

        )

    ########################################

    def forward(self,x):

        return self.network(x)

    ########################################

    def save(

        self,

        filename="models/flappy_model.pth"

    ):

        os.makedirs(

            "models",

            exist_ok=True

        )

        torch.save(

            self.state_dict(),

            filename

        )

    ########################################

    def load(

        self,

        filename="models/flappy_model.pth"

    ):

        self.load_state_dict(

            torch.load(filename)

        )

        self.eval()


########################################################


class Trainer:

    def __init__(

        self,

        model,

        lr=0.001,

        gamma=0.99

    ):

        self.model = model

        self.gamma = gamma

        self.optimizer = optim.Adam(

            model.parameters(),

            lr=lr

        )

        self.loss = nn.MSELoss()

    ####################################################

    def train_step(

        self,

        state,

        action,

        reward,

        next_state,

        done

    ):

        state = torch.tensor(

            state,

            dtype=torch.float

        )

        next_state = torch.tensor(

            next_state,

            dtype=torch.float

        )

        action = torch.tensor(

            action,

            dtype=torch.long

        )

        reward = torch.tensor(

            reward,

            dtype=torch.float

        )

        if len(state.shape) == 1:

            state = state.unsqueeze(0)

            next_state = next_state.unsqueeze(0)

            action = action.unsqueeze(0)

            reward = reward.unsqueeze(0)

            done = (done,)

        prediction = self.model(state)

        target = prediction.clone()

        for i in range(len(done)):

            q = reward[i]

            if not done[i]:

                q = reward[i] + self.gamma * torch.max(

                    self.model(

                        next_state[i]

                    )

                )

            target[i][

                torch.argmax(action[i]).item()

            ] = q

        self.optimizer.zero_grad()

        loss = self.loss(

            prediction,

            target

        )

        loss.backward()

        self.optimizer.step()