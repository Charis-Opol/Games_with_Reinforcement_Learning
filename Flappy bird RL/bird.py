import pygame

from config import *

class Bird:

    def __init__(self,x,y):

        self.x = x

        self.y = y

        self.velocity = 0

        self.rotation = 0
        # frames since last jump (used to enforce jump cooldown)
        self.frames_since_jump = JUMP_COOLDOWN_FRAMES

    #######################################

    def jump(self):

        # enforce a short cooldown to prevent jump-spam
        if self.frames_since_jump < JUMP_COOLDOWN_FRAMES:
            return

        self.velocity = JUMP_STRENGTH
        self.frames_since_jump = 0

    #######################################

    def update(self):

        self.velocity += GRAVITY

        if self.velocity > MAX_FALL_SPEED:

            self.velocity = MAX_FALL_SPEED

        self.y += self.velocity

        # increment jump cooldown counter
        self.frames_since_jump += 1

        if self.velocity < 0:

            self.rotation = MAX_UP_ROTATION

        else:

            self.rotation -= ROTATION_SPEED

            if self.rotation < MAX_DOWN_ROTATION:

                self.rotation = MAX_DOWN_ROTATION

    #######################################

    def draw(self,screen):

        bird_surface = pygame.Surface(
            (50,50),
            pygame.SRCALPHA
        )

        pygame.draw.circle(

            bird_surface,

            YELLOW,

            (25,25),

            BIRD_RADIUS

        )

        pygame.draw.circle(

            bird_surface,

            BLACK,

            (32,20),

            3

        )

        pygame.draw.polygon(

            bird_surface,

            ORANGE,

            [

                (40,25),

                (50,22),

                (50,28)

            ]

        )

        rotated = pygame.transform.rotate(

            bird_surface,

            self.rotation

        )

        rect = rotated.get_rect(

            center=(self.x,self.y)

        )

        screen.blit(

            rotated,

            rect

        )
    
    def out_of_bounds(self):

        if self.y - BIRD_RADIUS <= 0:
            return True

        if self.y + BIRD_RADIUS >= HEIGHT - GROUND_HEIGHT:
            return True

        return False