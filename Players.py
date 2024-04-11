import pygame
import header

class Player(pygame.sprite.Sprite):
    def __init__(self, x, color):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20, 150))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.center = (x, header.HEIGHT - self.rect.height * 1.5)
        self.score = 0
        self.predictY = 0
        self.action = "nothing"
        
        
    def moveUp(self):
        self.rect.y -= 10

    def moveDown(self):
        self.rect.y += 10

    def update():
        pass
