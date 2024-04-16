import pygame
import random
import os

from header import *

class Ball(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20, 20), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (200,200,200), (10, 10), 10)
        self.rect = self.image.get_rect()

        self.maxSpeed = 7
        self.restartBall()

        self.player1 = 0
        self.player2 = 0

        self.SFX_TouchPlayer = pygame.mixer.Sound("SFX\TouchPlayer.mp3")
        self.SFX_TouchWall = pygame.mixer.Sound("SFX\TouchWall.mp3")

        self.SFX_TouchPlayer.set_volume(volumeSound)
        self.SFX_TouchWall.set_volume(volumeSound)
        


    def TouchPlayer(self):
        if self.rect.colliderect(self.player1.rect):
            bias = (self.rect.centery - self.player1.rect.centery) / self.player1.rect.height * self.maxSpeed * 1.5
            self.dy = bias
            self.dx = self.maxSpeed - abs(self.dy)
            self.maxSpeed += 0.0001
            self.SFX_TouchPlayer.play()

        if self.rect.colliderect(self.player2.rect):
            bias = (self.rect.centery - self.player2.rect.centery) / self.player2.rect.height * self.maxSpeed * 1.5
            self.dy = bias
            self.dx = abs(self.dy) - self.maxSpeed
            self.maxSpeed += 0.0001
            self.SFX_TouchPlayer.play()


    def TouchWall(self):
        if self.rect.right >= WIDTH and self.dx > 0:
            self.restartBall()
            self.player1.score+=1
            self.SFX_TouchWall.play()
            
        if self.rect.left <= 0 and self.dx < 0:
            self.restartBall()
            self.player2.score+=1
            self.SFX_TouchWall.play()
            
        if self.rect.top <= 0:
            self.dy = abs(self.dy)
            self.SFX_TouchWall.play()
        elif self.rect.bottom >= HEIGHT:
            self.dy = -abs(self.dy)
            self.SFX_TouchWall.play()

    def restartBall(self):
        self.rect.center = (WIDTH /2, HEIGHT / 2)

        self.dx = 0
        self.dy = 0

        self.max_dx = int(self.maxSpeed)
        self.min_dx = int(-self.max_dx)
        
        while -1 < self.dx < 1:
            self.dx = random.randint(self.min_dx, self.max_dx)
            self.max_dy = int(self.maxSpeed - abs(self.dx))
            self.min_dy = int(-self.max_dy)
            self.dy = random.randint(self.min_dy, self.max_dy)
            
            


    def update(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        
        self.TouchPlayer()
        self.TouchWall()
        

        self.rect.x += self.dx
        self.rect.y += self.dy
