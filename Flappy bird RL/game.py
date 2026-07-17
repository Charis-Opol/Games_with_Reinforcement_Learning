import pygame
import random

from bird import Bird
from pipe import Pipe
from base import Base
from config import *
 
class FlappyBirdGame:

    def __init__(self):

        pygame.init()

        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT)
        )

        pygame.display.set_caption(
            "Flappy Bird RL"
        )

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(
            "Arial",
            FONT_SIZE
        )

        self.reset()

    ###################################################

    def reset(self):

        self.bird = Bird(

            WIDTH // 4,

            HEIGHT // 2

        )

        self.base = Base()

        self.pipes = [

            Pipe(WIDTH + 100)

        ]

        self.score = 0

        self.running = True

    ###################################################

    def update(self, action=None):

        self.clock.tick(FPS)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                pygame.quit()

                quit()

    # Human controls
        if action is None:

            keys = pygame.key.get_pressed()

            if keys[pygame.K_SPACE]:

                self.bird.jump()

    # AI controls
        else:

            # Normalize different possible action types (int, list/tuple one-hot,
            # numpy array, torch tensor) to a single integer 0 or 1.
            ai_action = 0

            # torch.Tensor
            try:
                import torch

                if isinstance(action, torch.Tensor):
                    if action.dim() == 0:
                        ai_action = int(action.item())
                    else:
                        ai_action = int(action.argmax().item())
                else:
                    raise Exception()
            except Exception:
                # numpy array
                try:
                    import numpy as np

                    if isinstance(action, np.ndarray):
                        if action.ndim == 0:
                            ai_action = int(action.item())
                        else:
                            ai_action = int(action.argmax())
                    else:
                        raise Exception()
                except Exception:
                    # list/tuple or plain int-like
                    if isinstance(action, (list, tuple)):
                        try:
                            if len(action) == 2:
                                # assume one-hot [1,0] or [0,1]
                                ai_action = int(action.index(max(action)))
                            else:
                                ai_action = int(action[0])
                        except Exception:
                            try:
                                ai_action = int(action)
                            except Exception:
                                ai_action = 0
                    else:
                        try:
                            ai_action = int(action)
                        except Exception:
                            ai_action = 0

            # optional rule-based assistance
            if ASSISTED_MODE and random.random() < ASSIST_PROBABILITY:
                ai_action = self.assist_action(ai_action)

            if ai_action == 1:
                self.bird.jump()

        self.bird.update()

        self.base.update()

        # check for ceiling or ground collisions
        if self.bird.out_of_bounds():
            self.running = False

        add_pipe = False

        for pipe in self.pipes[:]:

            pipe.update()

            if pipe.collide(self.bird):

                self.running = False

            if not pipe.passed and pipe.x < self.bird.x:

                pipe.passed = True

                add_pipe = True

                self.score += 1

            if pipe.off_screen():

                self.pipes.remove(pipe)

        if add_pipe:

            self.pipes.append(Pipe(WIDTH))
    ###################################################

    def draw(self):

        self.screen.fill(BLUE)

        for pipe in self.pipes:

            pipe.draw(self.screen)

        self.base.draw(self.screen)

        self.bird.draw(self.screen)

        score = self.font.render(

            f"Score: {self.score}",

            True,

            TEXT_COLOR

        )

        self.screen.blit(

            score,

            (20,20)

        )

        pygame.display.flip()

    ###################################################

    def play_step(self, action=None):

        self.update(action)

        self.draw()

        reward = 0.1

        done = False

        if not self.running:

            reward = -10

            done = True

        if self.score > 0:

            reward = 10

        # real-time position-based reward shaping (vertical + horizontal closeness)
        try:
            next_pipe = self.pipes[0]
            if len(self.pipes) > 1 and self.pipes[0].x < self.bird.x:
                next_pipe = self.pipes[1]

            gap_center_y = next_pipe.height + PIPE_GAP / 2
            gap_center_x = next_pipe.x + PIPE_WIDTH / 2

            v_dist = abs(self.bird.y - gap_center_y)
            h_dist = abs(self.bird.x - gap_center_x)

            v_score = max(0.0, 1.0 - (v_dist / HEIGHT)) * SHAPING_VERTICAL_WEIGHT
            h_score = max(0.0, 1.0 - (h_dist / WIDTH)) * SHAPING_HORIZONTAL_WEIGHT

            shaping = v_score + h_score
            reward += shaping
        except Exception:
            pass

        return reward, done, self.score
    
    def get_state(self):

        next_pipe = self.pipes[0]

        if len(self.pipes) > 1 and self.pipes[0].x < self.bird.x:

            next_pipe = self.pipes[1]

        return [
            self.bird.y,
            self.bird.velocity,
            next_pipe.x - self.bird.x,
            next_pipe.height,
            next_pipe.height + PIPE_GAP

    ]

    ###################################
    def assist_action(self, current_action):
        """Simple rule-based corrective action: if bird is below gap center by more than
        ASSIST_MARGIN pixels, request a jump. Otherwise keep current action."""
        try:
            # determine the next relevant pipe
            next_pipe = self.pipes[0]
            if len(self.pipes) > 1 and self.pipes[0].x < self.bird.x:
                next_pipe = self.pipes[1]

            gap_center = next_pipe.height + PIPE_GAP / 2
            if self.bird.y > gap_center + ASSIST_MARGIN:
                return 1
            if self.bird.y < gap_center - ASSIST_MARGIN:
                return 0
        except Exception:
            pass

        return current_action