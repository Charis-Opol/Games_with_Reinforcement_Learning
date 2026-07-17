import random
import pygame

from config import *


class Pipe:

    def __init__(self, x):

        self.x = x

        self.height = random.randint(

            PIPE_MIN_HEIGHT,

            PIPE_MAX_HEIGHT

        )

        self.passed = False

    #####################################

    def update(self):

        self.x -= PIPE_SPEED

    #####################################

    def off_screen(self):

        return self.x + PIPE_WIDTH < 0

    #####################################

    def collide(self, bird):

        bird_left = bird.x - BIRD_RADIUS
        bird_right = bird.x + BIRD_RADIUS
        bird_top = bird.y - BIRD_RADIUS
        bird_bottom = bird.y + BIRD_RADIUS

        top_rect = pygame.Rect(

            self.x,

            0,

            PIPE_WIDTH,

            self.height

        )

        bottom_rect = pygame.Rect(

            self.x,

            self.height + PIPE_GAP,

            PIPE_WIDTH,

            HEIGHT - self.height - PIPE_GAP - GROUND_HEIGHT

        )

        bird_rect = pygame.Rect(
            bird_left,
            bird_top,
            BIRD_RADIUS * 2,
            BIRD_RADIUS * 2,
        )

        return top_rect.colliderect(bird_rect) or bottom_rect.colliderect(bird_rect)

    #####################################

    def draw(self, screen):

        pygame.draw.rect(

            screen,

            PIPE_COLOR,

            (

                self.x,

                0,

                PIPE_WIDTH,

                self.height

            )

        )

        pygame.draw.rect(

            screen,

            PIPE_COLOR,

            (

                self.x,

                self.height + PIPE_GAP,

                PIPE_WIDTH,

                HEIGHT - self.height - PIPE_GAP - GROUND_HEIGHT

            )

        )