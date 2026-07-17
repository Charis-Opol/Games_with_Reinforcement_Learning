from model import DQN

import torch

model = DQN()

x = torch.rand(5)

print(model(x))

from replay_memory import ReplayMemory

memory = ReplayMemory()

for i in range(100):

    memory.push(

        i,

        0,

        1,

        i+1,

        False

    )

print(

    len(memory)

)

states,actions,rewards,next_states,dones = memory.sample(10)

print(list(states))