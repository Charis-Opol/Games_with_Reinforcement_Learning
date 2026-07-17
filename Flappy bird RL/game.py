import pygame

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

            if action == 1:

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