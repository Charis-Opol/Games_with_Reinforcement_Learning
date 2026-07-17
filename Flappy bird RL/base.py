import pygame

from config import *


class Base:

    def __init__(self):

        self.x1 = 0

        self.x2 = WIDTH

    ########################################

    def update(self):

        self.x1 -= GROUND_SPEED

        self.x2 -= GROUND_SPEED

        if self.x1 + WIDTH <= 0:

            self.x1 = self.x2 + WIDTH

        if self.x2 + WIDTH <= 0:

            self.x2 = self.x1 + WIDTH

    ########################################

    def draw(self, screen):

        pygame.draw.rect(

            screen,

            GROUND_COLOR,

            (self.x1,
             HEIGHT - GROUND_HEIGHT,
             WIDTH,
             GROUND_HEIGHT)

        )

        pygame.draw.rect(

            screen,

            GROUND_COLOR,

            (self.x2,
             HEIGHT - GROUND_HEIGHT,
             WIDTH,
             GROUND_HEIGHT)

        )