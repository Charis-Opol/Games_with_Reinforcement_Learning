import random
import numpy as np
import torch

from replay_memory import ReplayMemory
from model import DQN, Trainer


MAX_MEMORY = 100000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class Agent:

    ########################################################

    def __init__(self):

        self.games = 0

        self.gamma = 0.99

        self.epsilon = 1.0

        self.epsilon_decay = 0.995

        self.epsilon_min = 0.01

        self.memory = ReplayMemory(MAX_MEMORY)

        self.model = DQN()

        self.trainer = Trainer(

            self.model,

            lr=LEARNING_RATE,

            gamma=self.gamma

        )

    ########################################################

    def get_state(self, game):

        return np.array(

            game.get_state(),

            dtype=np.float32

        )

    ########################################################

    def remember(

        self,

        state,

        action,

        reward,

        next_state,

        done

    ):

        self.memory.push(

            state,

            action,

            reward,

            next_state,

            done

        )

    ########################################################

    def train_short_memory(

        self,

        state,

        action,

        reward,

        next_state,

        done

    ):

        action_vector = [0, 0]

        action_vector[action] = 1

        self.trainer.train_step(

            state,

            action_vector,

            reward,

            next_state,

            done

        )

    ########################################################

    def train_long_memory(self):

        if len(self.memory) == 0:

            return

        batch_size = min(

            BATCH_SIZE,

            len(self.memory)

        )

        states, actions, rewards, next_states, dones = self.memory.sample(

            batch_size

        )

        actions = [

            [1, 0] if a == 0 else [0, 1]

            for a in actions

        ]

        self.trainer.train_step(

            states,

            actions,

            rewards,

            next_states,

            dones

        )

    ########################################################

    def get_action(

        self,

        state

    ):

        if random.random() < self.epsilon:

            return random.randint(

                0,

                1

            )

        state = torch.tensor(

            state,

            dtype=torch.float

        )

        prediction = self.model(

            state

        )

        return torch.argmax(

            prediction

        ).item()

    ########################################################

    def end_game(self):

        self.games += 1

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay