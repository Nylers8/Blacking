import pygame

from header import *

class CalBall(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (200,200,200), (10, 10), 10)
        self.rect = self.image.get_rect()

        self.maxSpeed = 7

        self.player1 = 0
        self.player2 = 0

        


    def TouchPlayer(self):
        if self.rect.colliderect(self.player1.rect):
            bias = (self.rect.centery - self.player1.rect.centery) / self.player1.rect.height * self.maxSpeed * 1.5
            self.dy = bias
            self.dx = self.maxSpeed - abs(self.dy)
            self.maxSpeed += 0.05


        if self.rect.colliderect(self.player2.rect):
            bias = (self.rect.centery - self.player2.rect.centery) / self.player2.rect.height * self.maxSpeed * 1.5
            self.dy = bias
            self.dx = abs(self.dy) - self.maxSpeed
            self.maxSpeed += 0.05


    def TouchWall(self):
            
        if self.rect.top <= 0:
            self.dy = abs(self.dy)
        elif self.rect.bottom >= HEIGHT:
            self.dy = -abs(self.dy)
            

    def update(self):
        self.TouchWall()

        self.rect.x += self.dx
        self.rect.y += self.dy
